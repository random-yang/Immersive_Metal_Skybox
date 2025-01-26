// //
// //  Shaders.metal
// //

// // File for Metal kernel and shader functions

// #include <metal_stdlib>
// #include <simd/simd.h>

// // Including header shared between this Metal shader code and Swift/C code executing Metal API commands
// #import "ShaderTypes.h"

// using namespace metal;

// typedef struct
// {
//     float3 position [[attribute(VertexAttributePosition)]];
//     float2 texCoord [[attribute(VertexAttributeTexcoord)]];
// } Vertex;

// typedef struct
// {
//     float4 position [[position]];
//     float2 texCoord;
// } ColorInOut;

// vertex ColorInOut vertexShader(Vertex in [[stage_in]],
//                                ushort amp_id [[amplification_id]],
//                                constant UniformsArray & uniformsArray [[ buffer(BufferIndexUniforms) ]])
// {
//     ColorInOut out;

//     Uniforms uniforms = uniformsArray.uniforms[amp_id];
    
//     float4 position = float4(in.position, 1.0);
//     out.position = uniforms.projectionMatrix * uniforms.modelViewMatrix * position;
//     out.texCoord = in.texCoord;

//     return out;
// }

// fragment float4 fragmentShader(ColorInOut in [[stage_in]],
//                                texture2d<half> colorMap     [[ texture(TextureIndexColor) ]])
// {
//     constexpr sampler colorSampler(mip_filter::linear,
//                                    mag_filter::linear,
//                                    min_filter::linear);

//     half4 colorSample   = colorMap.sample(colorSampler, in.texCoord.xy);

//     return float4(colorSample);
// }

//
//  shader.metal
//  Metal_skybox
//
//  Created by randomyang on 2025/1/22.
//

#include <metal_stdlib>
#import "ShaderTypes.h"
using namespace metal;

struct VertexIn {
    float3 position [[attribute(0)]];
};

struct VertexOut {
    float4 position [[position]];
    float3 texCoord;
};

vertex VertexOut skyboxVertex(
                              VertexIn in [[stage_in]],
                              ushort amp_id [[amplification_id]],
                              constant UniformsArray & uniformsArray [[ buffer(BufferIndexUniforms) ]]
) {
    VertexOut out;
    Uniforms uniforms = uniformsArray.uniforms[amp_id];
    out.position = uniforms.projectionMatrix * uniforms.modelViewMatrix * float4(in.position, 1.0);
    out.texCoord = in.position; // 使用顶点位置作为采样方向
    return out;
}

fragment float4 skyboxFragment(
    VertexOut in [[stage_in]],
    texturecube<float> cubeTexture [[texture(0)]]
) {
    constexpr sampler cubeSampler(filter::linear, mip_filter::linear);
    return cubeTexture.sample(cubeSampler, in.texCoord);
}
