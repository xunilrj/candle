@group(0) @binding(0) var<storage, read> l: array<f32>;
@group(0) @binding(1) var<storage, read> r: array<f32>;

@group(1) @binding(0) var<storage, read_write> o: array<f32>;

@compute @workgroup_size(1) fn kernel_add(
    @builtin(global_invocation_id) id: vec3<u32>
) {
    let i = id.x;
    o[i] = l[i] + r[i];
}