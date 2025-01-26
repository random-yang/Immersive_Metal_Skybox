//
//  Immersive_Metal_SkyboxApp.swift
//  Immersive_Metal_Skybox
//
//  Created by randomyang on 2025/1/24.
//

import SwiftUI
import CompositorServices

struct ContentStageConfiguration: CompositorLayerConfiguration {
    func makeConfiguration(capabilities: LayerRenderer.Capabilities, configuration: inout LayerRenderer.Configuration) {
        configuration.depthFormat = .depth32Float
        configuration.colorFormat = .bgra8Unorm_srgb

        let foveationEnabled = capabilities.supportsFoveation
        configuration.isFoveationEnabled = foveationEnabled

        let options: LayerRenderer.Capabilities.SupportedLayoutsOptions = foveationEnabled ? [.foveationEnabled] : []
        let supportedLayouts = capabilities.supportedLayouts(options: options)

        configuration.layout = supportedLayouts.contains(.layered) ? .layered : .dedicated
    }
}

@main
struct Immersive_Metal_SkyboxTestApp: App {

    @State private var appModel = AppModel()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environment(appModel)
        }

        ImmersiveSpace(id: appModel.immersiveSpaceID) {
            // 以回调的方式提供上下文和函数返回的方式的区别
            // 回调函数：控制权在 api 内
            // 返回函数：控制权在 caller
            CompositorLayer(configuration: ContentStageConfiguration()) { @MainActor layerRenderer in
                Renderer.startRenderLoop(layerRenderer, appModel: appModel)
            }
        }
        .immersionStyle(selection: .constant(.full), in: .full)
    }
}

