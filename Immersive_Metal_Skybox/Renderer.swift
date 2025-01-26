//
//  Renderer.swift
//  Immersive_Metal_Skybox
//
//  Created by randomyang on 2025/1/24.
//

import CompositorServices
import Metal
import MetalKit
import simd
import Spatial

// The 256 byte aligned size of our uniform structure
let alignedUniformsSize = (MemoryLayout<UniformsArray>.size + 0xFF) & -0x100

let maxBuffersInFlight = 3

enum RendererError: Error {
    case badVertexDescriptor
}

extension LayerRenderer.Clock.Instant.Duration {
    var timeInterval: TimeInterval {
        let nanoseconds = TimeInterval(components.attoseconds / 1_000_000_000)
        return TimeInterval(components.seconds) + (nanoseconds / TimeInterval(NSEC_PER_SEC))
    }
}

final class RendererTaskExecutor: TaskExecutor {
    private let queue = DispatchQueue(label: "RenderThreadQueue", qos: .userInteractive)

    func enqueue(_ job: UnownedJob) {
        queue.async {
          job.runSynchronously(on: self.asUnownedSerialExecutor())
        }
    }

    func asUnownedSerialExecutor() -> UnownedTaskExecutor {
        return UnownedTaskExecutor(ordinary: self)
    }

    static var shared: RendererTaskExecutor = RendererTaskExecutor()
}

actor Renderer {

    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    var dynamicUniformBuffer: MTLBuffer
    var pipelineState: MTLRenderPipelineState
    var depthState: MTLDepthStencilState

    let inFlightSemaphore = DispatchSemaphore(value: maxBuffersInFlight)

    var uniformBufferOffset = 0

    var uniformBufferIndex = 0

    var uniforms: UnsafeMutablePointer<UniformsArray>

    let rasterSampleCount: Int
    var memorylessTargetIndex: Int = 0
    var memorylessTargets: [(color: MTLTexture, depth: MTLTexture)?]

    let arSession: ARKitSession
    let worldTracking: WorldTrackingProvider
    let layerRenderer: LayerRenderer
    let appModel: AppModel

    var skyboxVertexBuffer: MTLBuffer!
    var skyboxTexture: MTLTexture!

    init(_ layerRenderer: LayerRenderer, appModel: AppModel) {
        self.layerRenderer = layerRenderer
        self.device = layerRenderer.device
        self.commandQueue = self.device.makeCommandQueue()!
        self.appModel = appModel

        let device = self.device
        if device.supports32BitMSAA && device.supportsTextureSampleCount(4) {
            rasterSampleCount = 4
        } else {
            rasterSampleCount = 1
        }

        let uniformBufferSize = alignedUniformsSize * maxBuffersInFlight

        self.dynamicUniformBuffer = self.device.makeBuffer(length:uniformBufferSize,
                                                           options:[MTLResourceOptions.storageModeShared])!

        self.dynamicUniformBuffer.label = "UniformBuffer"

        self.memorylessTargets = .init(repeating: nil, count: maxBuffersInFlight)

        uniforms = UnsafeMutableRawPointer(dynamicUniformBuffer.contents()).bindMemory(to:UniformsArray.self, capacity:1)

        let mtlVertexDescriptor = Renderer.buildMetalVertexDescriptor()

        do {
            pipelineState = try Renderer.buildRenderPipelineWithDevice(device: device,
                                                                       layerRenderer: layerRenderer,
                                                                       rasterSampleCount: rasterSampleCount,
                                                                       mtlVertexDescriptor: mtlVertexDescriptor)
        } catch {
            fatalError("Unable to compile render pipeline state.  Error info: \(error)")
        }

        let depthStateDescriptor = MTLDepthStencilDescriptor()
        depthStateDescriptor.depthCompareFunction = MTLCompareFunction.greater
        depthStateDescriptor.isDepthWriteEnabled = true
        self.depthState = device.makeDepthStencilState(descriptor:depthStateDescriptor)!

        worldTracking = WorldTrackingProvider()
        arSession = ARKitSession()

        // 创建天空盒顶点数据
        let skyboxVertices: [SIMD3<Float>] = [
            // 前面
            [-100,  100,  100], [ 100,  100,  100], [-100, -100,  100],
            [ 100,  100,  100], [ 100, -100,  100], [-100, -100,  100],
            // 右面
            [ 100,  100,  100], [ 100,  100, -100], [ 100, -100,  100],
            [ 100,  100, -100], [ 100, -100, -100], [ 100, -100,  100],
            // 后面
            [ 100,  100, -100], [-100,  100, -100], [ 100, -100, -100],
            [-100,  100, -100], [-100, -100, -100], [ 100, -100, -100],
            // 左面
            [-100,  100, -100], [-100,  100,  100], [-100, -100, -100],
            [-100,  100,  100], [-100, -100,  100], [-100, -100, -100],
            // 顶面
            [-100,  100, -100], [ 100,  100, -100], [-100,  100,  100],
            [ 100,  100, -100], [ 100,  100,  100], [-100,  100,  100],
            // 底面
            [-100, -100,  100], [ 100, -100,  100], [-100, -100, -100],
            [ 100, -100,  100], [ 100, -100, -100], [-100, -100, -100]
        ]
        
        skyboxVertexBuffer = device.makeBuffer(
            bytes: skyboxVertices,
            length: skyboxVertices.count * MemoryLayout<SIMD3<Float>>.stride,
            options: .storageModeShared
        )

        // 加载天空盒纹理
        let textureLoader = MTKTextureLoader(device: device)
        skyboxTexture = try? textureLoader.newTexture(
            name: "Skybox",
            scaleFactor: 1.0,
            bundle: nil,
            options: [
                .textureUsage: NSNumber(value: MTLTextureUsage.shaderRead.rawValue),
                .textureStorageMode: NSNumber(value: MTLStorageMode.private.rawValue)
            ]
        )
    }

    private func startARSession() async {
        do {
            try await arSession.run([worldTracking])
        } catch {
            fatalError("Failed to initialize ARSession")
        }
    }

    @MainActor
    static func startRenderLoop(_ layerRenderer: LayerRenderer, appModel: AppModel) {
        Task(executorPreference: RendererTaskExecutor.shared) {
            let renderer = Renderer(layerRenderer, appModel: appModel)
            await renderer.startARSession()
            await renderer.renderLoop()
        }
    }

    static private func buildMetalVertexDescriptor() -> MTLVertexDescriptor {
        let vertexDescriptor = MTLVertexDescriptor()
        
        // 位置属性
        vertexDescriptor.attributes[VertexAttribute.position.rawValue].format = .float3
        vertexDescriptor.attributes[VertexAttribute.position.rawValue].offset = 0
        vertexDescriptor.attributes[VertexAttribute.position.rawValue].bufferIndex = BufferIndex.meshPositions.rawValue
        
        // 布局
        vertexDescriptor.layouts[BufferIndex.meshPositions.rawValue].stride = MemoryLayout<SIMD3<Float>>.stride
        vertexDescriptor.layouts[BufferIndex.meshPositions.rawValue].stepRate = 1
        vertexDescriptor.layouts[BufferIndex.meshPositions.rawValue].stepFunction = .perVertex
        
        return vertexDescriptor
    }

    static func buildRenderPipelineWithDevice(device: MTLDevice,
                                              layerRenderer: LayerRenderer,
                                              rasterSampleCount: Int,
                                              mtlVertexDescriptor: MTLVertexDescriptor) throws -> MTLRenderPipelineState {
        /// Build a render state pipeline object

        let library = device.makeDefaultLibrary()

        let vertexFunction = library?.makeFunction(name: "skyboxVertex")
        let fragmentFunction = library?.makeFunction(name: "skyboxFragment")

        let pipelineDescriptor = MTLRenderPipelineDescriptor()
        pipelineDescriptor.label = "RenderPipeline"
        pipelineDescriptor.vertexFunction = vertexFunction
        pipelineDescriptor.fragmentFunction = fragmentFunction
        pipelineDescriptor.vertexDescriptor = mtlVertexDescriptor
        pipelineDescriptor.rasterSampleCount = rasterSampleCount

        pipelineDescriptor.colorAttachments[0].pixelFormat = layerRenderer.configuration.colorFormat
        pipelineDescriptor.depthAttachmentPixelFormat = layerRenderer.configuration.depthFormat

        pipelineDescriptor.maxVertexAmplificationCount = layerRenderer.properties.viewCount

        return try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
    }

    private func updateDynamicBufferState() {
        /// Update the state of our uniform buffers before rendering

        uniformBufferIndex = (uniformBufferIndex + 1) % maxBuffersInFlight

        uniformBufferOffset = alignedUniformsSize * uniformBufferIndex

        uniforms = UnsafeMutableRawPointer(dynamicUniformBuffer.contents() + uniformBufferOffset).bindMemory(to:UniformsArray.self, capacity:1)
    }

    private func memorylessRenderTargets(drawable: LayerRenderer.Drawable) -> (color: MTLTexture, depth: MTLTexture) {

        func renderTarget(resolveTexture: MTLTexture, cachedTexture: MTLTexture?) -> MTLTexture {
            if let cachedTexture,
               resolveTexture.width == cachedTexture.width && resolveTexture.height == cachedTexture.height {
                return cachedTexture
            } else {
                let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: resolveTexture.pixelFormat,
                                                                          width: resolveTexture.width,
                                                                          height: resolveTexture.height,
                                                                          mipmapped: false)
                descriptor.usage = .renderTarget
                descriptor.textureType = .type2DMultisampleArray
                descriptor.sampleCount = rasterSampleCount
                descriptor.storageMode = .memoryless
                descriptor.arrayLength = resolveTexture.arrayLength
                return resolveTexture.device.makeTexture(descriptor: descriptor)!
            }
        }

        memorylessTargetIndex = (memorylessTargetIndex + 1) % maxBuffersInFlight

        let cachedTargets = memorylessTargets[memorylessTargetIndex]
        let newTargets = (renderTarget(resolveTexture: drawable.colorTextures[0], cachedTexture: cachedTargets?.color),
                          renderTarget(resolveTexture: drawable.depthTextures[0], cachedTexture: cachedTargets?.depth))

        memorylessTargets[memorylessTargetIndex] = newTargets

        return newTargets
    }

    private func updateGameState(drawable: LayerRenderer.Drawable, deviceAnchor: DeviceAnchor?) {
        /// Update any game state before rendering

        let simdDeviceAnchor = deviceAnchor?.originFromAnchorTransform ?? matrix_identity_float4x4

        func uniforms(forViewIndex viewIndex: Int) -> Uniforms {
            let view = drawable.views[viewIndex]
            let viewMatrix = (simdDeviceAnchor * view.transform).inverse
            let projection = drawable.computeProjection(viewIndex: viewIndex)
            let skyboxViewMatrix = viewMatrix

            return Uniforms(projectionMatrix: projection, modelViewMatrix: skyboxViewMatrix)
        }

        self.uniforms[0].uniforms.0 = uniforms(forViewIndex: 0)
        if drawable.views.count > 1 {
            self.uniforms[0].uniforms.1 = uniforms(forViewIndex: 1)
        }
    }

    func renderFrame() {
        /// Per frame updates hare

        guard let frame = layerRenderer.queryNextFrame() else { return }

        frame.startUpdate()

        // Perform frame independent work

        frame.endUpdate()

        guard let timing = frame.predictTiming() else { return }
        LayerRenderer.Clock().wait(until: timing.optimalInputTime)

        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            fatalError("Failed to create command buffer")
        }

        guard let drawable = frame.queryDrawable() else { return }

        _ = inFlightSemaphore.wait(timeout: DispatchTime.distantFuture)

        frame.startSubmission()

        let time = LayerRenderer.Clock.Instant.epoch.duration(to: drawable.frameTiming.presentationTime).timeInterval
        let deviceAnchor = worldTracking.queryDeviceAnchor(atTimestamp: time)

        drawable.deviceAnchor = deviceAnchor

        let semaphore = inFlightSemaphore
        commandBuffer.addCompletedHandler { (_ commandBuffer)-> Swift.Void in
            semaphore.signal()
        }

        self.updateDynamicBufferState()

        self.updateGameState(drawable: drawable, deviceAnchor: deviceAnchor)
 
        let renderPassDescriptor = MTLRenderPassDescriptor()

         if rasterSampleCount > 1 {
             let renderTargets = memorylessRenderTargets(drawable: drawable)
             renderPassDescriptor.colorAttachments[0].resolveTexture = drawable.colorTextures[0]
             renderPassDescriptor.colorAttachments[0].texture = renderTargets.color
             renderPassDescriptor.depthAttachment.resolveTexture = drawable.depthTextures[0]
             renderPassDescriptor.depthAttachment.texture = renderTargets.depth

             renderPassDescriptor.colorAttachments[0].storeAction = .multisampleResolve
             renderPassDescriptor.depthAttachment.storeAction = .multisampleResolve
         } else {
            renderPassDescriptor.colorAttachments[0].texture = drawable.colorTextures[0]
            renderPassDescriptor.depthAttachment.texture = drawable.depthTextures[0]

            renderPassDescriptor.colorAttachments[0].storeAction = .store
            renderPassDescriptor.depthAttachment.storeAction = .store
         }

        renderPassDescriptor.colorAttachments[0].loadAction = .clear
        renderPassDescriptor.colorAttachments[0].clearColor = MTLClearColor(red: 0.0, green: 0.0, blue: 0.0, alpha: 0.0)
        renderPassDescriptor.depthAttachment.loadAction = .clear
        renderPassDescriptor.depthAttachment.clearDepth = 0.0
        renderPassDescriptor.rasterizationRateMap = drawable.rasterizationRateMaps.first
        if layerRenderer.configuration.layout == .layered {
            renderPassDescriptor.renderTargetArrayLength = drawable.views.count
        }

        /// Final pass rendering code here
        guard let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) else {
            fatalError("Failed to create render encoder")
        }

        renderEncoder.label = "Primary Render Encoder"

        // 绘制天空盒
        renderEncoder.pushDebugGroup("Draw Skybox")
        renderEncoder.setCullMode(.front)
        renderEncoder.setRenderPipelineState(pipelineState)
        renderEncoder.setDepthStencilState(depthState)
        renderEncoder.setVertexBuffer(skyboxVertexBuffer, offset: 0, index: BufferIndex.meshPositions.rawValue)
        renderEncoder.setVertexBuffer(dynamicUniformBuffer, offset: uniformBufferOffset, index: BufferIndex.uniforms.rawValue)
        renderEncoder.setFragmentTexture(skyboxTexture, index: TextureIndex.color.rawValue)
        
        let viewports = drawable.views.map { $0.textureMap.viewport }
        renderEncoder.setViewports(viewports)
        
        if drawable.views.count > 1 {
            var viewMappings = (0..<drawable.views.count).map {
                MTLVertexAmplificationViewMapping(viewportArrayIndexOffset: UInt32($0),
                                                  renderTargetArrayIndexOffset: UInt32($0))
            }
            renderEncoder.setVertexAmplificationCount(viewports.count, viewMappings: &viewMappings)
        }
        
        renderEncoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 36)
        renderEncoder.popDebugGroup()

        renderEncoder.endEncoding()

        drawable.encodePresent(commandBuffer: commandBuffer)

        commandBuffer.commit()

        frame.endSubmission()
    }

    func renderLoop() {
        while true {
            if layerRenderer.state == .invalidated {
                print("Layer is invalidated")
                Task { @MainActor in
                    appModel.immersiveSpaceState = .closed
                }
                return
            } else if layerRenderer.state == .paused {
                Task { @MainActor in
                    appModel.immersiveSpaceState = .inTransition
                }
                layerRenderer.waitUntilRunning()
                continue
            } else {
                Task { @MainActor in
                    if appModel.immersiveSpaceState != .open {
                        appModel.immersiveSpaceState = .open
                    }
                }
                autoreleasepool {
                    self.renderFrame()
                }
            }
        }
    }
}
