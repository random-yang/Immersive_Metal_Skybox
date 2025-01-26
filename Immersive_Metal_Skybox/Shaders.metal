//
//  shader.metal
//  Metal_skybox
//
//  Created by randomyang on 2025/1/26.
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
