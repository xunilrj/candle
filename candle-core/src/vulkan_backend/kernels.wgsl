@group(0) @binding(0) var<storage, read> l: array<f32>;
@group(0) @binding(1) var<storage, read> r: array<f32>;

@group(1) @binding(0) var<storage, read_write> o: array<f32>;

struct PushConstants {
    method: u32
}
var<push_constant> pc: PushConstants;

const KERNEL_ADD: u32 = 1;
const KERNEL_SUB: u32 = 2;

@compute @workgroup_size(1) fn main(
    @builtin(global_invocation_id) id: vec3<u32>
) {
    switch pc.method {
        case KERNEL_ADD: { kernel_add(id); }
        case KERNEL_SUB: { kernel_sub(id); }
        default: {}
    }
}

fn kernel_add(id: vec3<u32>) {
    let i = id.x;
    o[i] = l[i] + r[i];
}

fn kernel_sub(id: vec3<u32>) {
    let i = id.x;
    o[i] = l[i] - r[i];
}
