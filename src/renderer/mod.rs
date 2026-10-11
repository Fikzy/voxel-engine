use std::sync::Arc;

use anyhow::{Context, Result};
use bytemuck::{Pod, Zeroable};
use glam::Mat4;
use wgpu::{CurrentSurfaceTexture, util::DeviceExt};
use winit::{dpi::PhysicalSize, window::Window};

use crate::world::{CHUNK_EDGE, ChunkCoord, World};

const RAYCAST_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CameraUniform {
    view_projection: [[f32; 4]; 4],
    inverse_view_projection: [[f32; 4]; 4],
    camera_position: [f32; 4],
    viewport: [f32; 4],
    chunk_origin: [i32; 4],
    chunk_grid: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DebugVertex {
    position: [f32; 3],
    color: [f32; 3],
}

impl DebugVertex {
    const ATTRIBUTES: [wgpu::VertexAttribute; 2] =
        wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3];

    fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBUTES,
        }
    }
}

pub struct Renderer {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    raycast_pipeline: wgpu::ComputePipeline,
    composite_pipeline: wgpu::RenderPipeline,
    debug_pipeline: wgpu::RenderPipeline,
    voxel_buffer: wgpu::Buffer,
    _raycast_texture: wgpu::Texture,
    _raycast_view: wgpu::TextureView,
    camera_buffer: wgpu::Buffer,
    raycast_bind_group: wgpu::BindGroup,
    composite_bind_group: wgpu::BindGroup,
    debug_vertex_buffer: wgpu::Buffer,
    debug_vertex_count: u32,
    stream_origin: ChunkCoord,
    stream_dimensions: [u32; 3],
    debug_visualization: bool,
}

impl Renderer {
    pub async fn new(window: Arc<Window>, world: &World) -> Result<Self> {
        let size = window.inner_size();
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let surface = instance
            .create_surface(window)
            .context("failed to create surface")?;
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
                apply_limit_buckets: false,
            })
            .await
            .context("no compatible graphics adapter found")?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .context("failed to create graphics device")?;
        let config = surface
            .get_default_config(&adapter, size.width.max(1), size.height.max(1))
            .context("surface does not support a presentation configuration")?;
        surface.configure(&device, &config);

        let (stream_origin, stream_dimensions) = world
            .stream_bounds()
            .context("world must have streamed chunks before renderer initialization")?;
        let voxel_data = build_voxel_data(world, stream_origin, stream_dimensions);
        let voxel_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Streamed world voxels"),
            contents: bytemuck::cast_slice(&voxel_data),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera uniform buffer"),
            contents: bytemuck::bytes_of(&CameraUniform::zeroed()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let raycast_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Raycast compute bind group layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: RAYCAST_FORMAT,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                ],
            });
        let raycast_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Chunk raycast compute shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("raycast.wgsl").into()),
        });
        let raycast_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Chunk raycast compute pipeline layout"),
                bind_group_layouts: &[Some(&raycast_bind_group_layout)],
                immediate_size: 0,
            });
        let raycast_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Chunk raycast compute pipeline"),
            layout: Some(&raycast_pipeline_layout),
            module: &raycast_shader,
            entry_point: Some("raycast_image"),
            compilation_options: Default::default(),
            cache: None,
        });

        let mesh_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Composition and debug mesh shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("mesh.wgsl").into()),
        });
        let mesh_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Composition and debug mesh bind group layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let mesh_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Composition and debug mesh pipeline layout"),
            bind_group_layouts: &[Some(&mesh_bind_group_layout)],
            immediate_size: 0,
        });
        let composite_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Raycast image composition pipeline"),
            layout: Some(&mesh_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &mesh_shader,
                entry_point: Some("composite_vs"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &mesh_shader,
                entry_point: Some("composite_fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        let debug_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Chunk boundary debug pipeline"),
            layout: Some(&mesh_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &mesh_shader,
                entry_point: Some("debug_vs"),
                compilation_options: Default::default(),
                buffers: &[Some(DebugVertex::layout())],
            },
            fragment: Some(wgpu::FragmentState {
                module: &mesh_shader,
                entry_point: Some("debug_fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        let (raycast_texture, raycast_view) =
            create_raycast_output(&device, config.width, config.height);
        let raycast_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Chunk raycast compute bind group"),
            layout: &raycast_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: voxel_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&raycast_view),
                },
            ],
        });
        let composite_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Raycast composition and debug bind group"),
            layout: &mesh_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&raycast_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: camera_buffer.as_entire_binding(),
                },
            ],
        });
        let debug_vertices = create_chunk_boundary_vertices(stream_origin, stream_dimensions);
        let debug_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Chunk boundary debug vertices"),
            contents: bytemuck::cast_slice(&debug_vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        Ok(Self {
            surface,
            device,
            queue,
            config,
            raycast_pipeline,
            composite_pipeline,
            debug_pipeline,
            voxel_buffer,
            _raycast_texture: raycast_texture,
            _raycast_view: raycast_view,
            camera_buffer,
            raycast_bind_group,
            composite_bind_group,
            debug_vertex_buffer,
            debug_vertex_count: debug_vertices.len() as u32,
            stream_origin,
            stream_dimensions,
            debug_visualization: false,
        })
    }

    pub fn aspect_ratio(&self) -> f32 {
        self.config.width as f32 / self.config.height as f32
    }

    pub fn set_debug_visualization(&mut self, enabled: bool) {
        self.debug_visualization = enabled;
    }

    pub fn update_world(&mut self, world: &World) {
        let Some((origin, dimensions)) = world.stream_bounds() else {
            return;
        };
        if origin == self.stream_origin && dimensions == self.stream_dimensions {
            return;
        }

        let voxel_data = build_voxel_data(world, origin, dimensions);
        self.voxel_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Streamed world voxels"),
                contents: bytemuck::cast_slice(&voxel_data),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let (raycast_layout, mesh_layout) = (
            self.raycast_pipeline.get_bind_group_layout(0),
            self.composite_pipeline.get_bind_group_layout(0),
        );
        let raycast_view = self._raycast_view.clone();
        self.raycast_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Streamed world raycast binding"),
            layout: &raycast_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.voxel_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&raycast_view),
                },
            ],
        });
        self.composite_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Streamed world composition binding"),
            layout: &mesh_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&raycast_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.camera_buffer.as_entire_binding(),
                },
            ],
        });
        let debug_vertices = create_chunk_boundary_vertices(origin, dimensions);
        self.debug_vertex_buffer =
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Streamed chunk boundary vertices"),
                    contents: bytemuck::cast_slice(&debug_vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                });
        self.debug_vertex_count = debug_vertices.len() as u32;
        self.stream_origin = origin;
        self.stream_dimensions = dimensions;
    }

    pub fn resize(&mut self, size: PhysicalSize<u32>) {
        if size.width == 0 || size.height == 0 {
            return;
        }

        self.config.width = size.width;
        self.config.height = size.height;
        self.surface.configure(&self.device, &self.config);
        let (texture, view) = create_raycast_output(&self.device, size.width, size.height);
        let raycast_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Chunk raycast compute bind group"),
            layout: &self.raycast_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.voxel_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
            ],
        });
        let composite_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Raycast composition and debug bind group"),
            layout: &self.composite_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.camera_buffer.as_entire_binding(),
                },
            ],
        });
        self._raycast_texture = texture;
        self._raycast_view = view;
        self.raycast_bind_group = raycast_bind_group;
        self.composite_bind_group = composite_bind_group;
    }

    pub fn render(&mut self, view_projection: Mat4, camera_position: glam::Vec3) -> Result<()> {
        let frame = match self.surface.get_current_texture() {
            CurrentSurfaceTexture::Success(frame) | CurrentSurfaceTexture::Suboptimal(frame) => {
                frame
            }
            CurrentSurfaceTexture::Lost | CurrentSurfaceTexture::Outdated => {
                self.surface.configure(&self.device, &self.config);
                return Ok(());
            }
            CurrentSurfaceTexture::Timeout | CurrentSurfaceTexture::Occluded => return Ok(()),
            CurrentSurfaceTexture::Validation => {
                anyhow::bail!("surface frame acquisition failed validation")
            }
        };
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::bytes_of(&CameraUniform {
                view_projection: view_projection.to_cols_array_2d(),
                inverse_view_projection: view_projection.inverse().to_cols_array_2d(),
                camera_position: camera_position.extend(1.0).to_array(),
                viewport: [
                    self.config.width as f32,
                    self.config.height as f32,
                    0.0,
                    0.0,
                ],
                chunk_origin: [
                    self.stream_origin.x,
                    self.stream_origin.y,
                    self.stream_origin.z,
                    0,
                ],
                chunk_grid: [
                    self.stream_dimensions[0],
                    self.stream_dimensions[1],
                    self.stream_dimensions[2],
                    CHUNK_EDGE as u32,
                ],
            }),
        );

        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Frame encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Chunk raycast compute pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.raycast_pipeline);
            pass.set_bind_group(0, &self.raycast_bind_group, &[]);
            pass.dispatch_workgroups(
                self.config.width.div_ceil(8),
                self.config.height.div_ceil(8),
                1,
            );
        }
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Raycast image composition pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.055,
                            g: 0.067,
                            b: 0.08,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.composite_pipeline);
            pass.set_bind_group(0, &self.composite_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        if self.debug_visualization{
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Chunk boundary debug mesh pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.debug_pipeline);
            pass.set_bind_group(0, &self.composite_bind_group, &[]);
            pass.set_vertex_buffer(0, self.debug_vertex_buffer.slice(..));
            pass.draw(0..self.debug_vertex_count, 0..1);
        }

        self.queue.submit([encoder.finish()]);
        self.queue.present(frame);
        Ok(())
    }
}

fn create_raycast_output(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Compute raycast output image"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: RAYCAST_FORMAT,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

fn build_voxel_data(world: &World, origin: ChunkCoord, dimensions: [u32; 3]) -> Vec<u32> {
    let mut voxels =
        Vec::with_capacity(dimensions.iter().product::<u32>() as usize * CHUNK_EDGE.pow(3));
    for z in 0..dimensions[2] {
        for y in 0..dimensions[1] {
            for x in 0..dimensions[0] {
                let coord = ChunkCoord {
                    x: origin.x + x as i32,
                    y: origin.y + y as i32,
                    z: origin.z + z as i32,
                };
                if let Some(chunk) = world.chunk(coord) {
                    voxels.extend(chunk.voxels().iter().map(|&material| material as u32));
                } else {
                    voxels.resize(voxels.len() + CHUNK_EDGE.pow(3), 0);
                }
            }
        }
    }
    voxels
}

fn create_chunk_boundary_vertices(origin: ChunkCoord, dimensions: [u32; 3]) -> Vec<DebugVertex> {
    let edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ];
    let mut vertices = Vec::new();
    for z in 0..dimensions[2] {
        for y in 0..dimensions[1] {
            for x in 0..dimensions[0] {
                let min = [
                    (origin.x + x as i32) as f32 * CHUNK_EDGE as f32,
                    (origin.y + y as i32) as f32 * CHUNK_EDGE as f32,
                    (origin.z + z as i32) as f32 * CHUNK_EDGE as f32,
                ];
                let max = min.map(|axis| axis + CHUNK_EDGE as f32);
                let corners = [
                    [min[0], min[1], min[2]],
                    [max[0], min[1], min[2]],
                    [max[0], max[1], min[2]],
                    [min[0], max[1], min[2]],
                    [min[0], min[1], max[2]],
                    [max[0], min[1], max[2]],
                    [max[0], max[1], max[2]],
                    [min[0], max[1], max[2]],
                ];
                vertices.extend(edges.into_iter().flat_map(|(start, end)| {
                    [start, end].map(|corner| DebugVertex {
                        position: corners[corner],
                        color: [0.25, 0.9, 0.95],
                    })
                }));
            }
        }
    }
    vertices
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc;

    use bytemuck::{Pod, Zeroable};
    use glam::Vec3;
    use wgpu::util::DeviceExt;

    use super::{CameraUniform, RAYCAST_FORMAT, build_voxel_data, create_raycast_output};
    use crate::world::{CHUNK_EDGE, Chunk, ChunkCoord, World};

    #[repr(C)]
    #[derive(Clone, Copy, Pod, Zeroable)]
    struct RayProbe {
        origin: [f32; 4],
        direction: [f32; 4],
    }

    #[test]
    fn gpu_traversal_matches_cpu_hits_and_chunk_boundaries() {
        pollster::block_on(async {
            let instance =
                wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
            let Ok(adapter) = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::LowPower,
                    force_fallback_adapter: false,
                    compatible_surface: None,
                    apply_limit_buckets: false,
                })
                .await
            else {
                eprintln!("skipping GPU traversal parity check: no adapter available");
                return;
            };
            let Ok((device, queue)) = adapter
                .request_device(&wgpu::DeviceDescriptor::default())
                .await
            else {
                eprintln!("skipping GPU traversal parity check: device unavailable");
                return;
            };

            let edge = CHUNK_EDGE as i32;
            let middle = CHUNK_EDGE / 2;
            let mut world = World::default();
            world.stream_around(Vec3::splat(edge as f32), 1);
            let mut first_chunk = Chunk::new();
            first_chunk.set_voxel([middle, middle, middle], 1);
            world.insert_chunk(ChunkCoord { x: 0, y: 1, z: 1 }, first_chunk);
            let mut next_chunk = Chunk::new();
            next_chunk.set_voxel([1.min(CHUNK_EDGE - 1), middle, middle], 2);
            world.insert_chunk(ChunkCoord { x: 1, y: 1, z: 1 }, next_chunk);
            let (stream_origin, stream_dimensions) = world.stream_bounds().unwrap();
            let voxels = build_voxel_data(&world, stream_origin, stream_dimensions);
            let voxel_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("GPU traversal test voxels"),
                contents: bytemuck::cast_slice(&voxels),
                usage: wgpu::BufferUsages::STORAGE,
            });
            let ray_cases = [
                (
                    Vec3::new(
                        edge as f32 - 1.5,
                        edge as f32 + middle as f32 + 0.5,
                        edge as f32 + middle as f32 + 0.5,
                    ),
                    Vec3::X,
                ),
                (
                    Vec3::new(
                        edge as f32 + middle as f32 + 0.5,
                        edge as f32 + middle as f32 + 0.5,
                        edge as f32 + middle as f32 + 0.5,
                    ),
                    -Vec3::X,
                ),
                (
                    Vec3::new(
                        stream_dimensions[0] as f32 * edge as f32 - 0.5,
                        edge as f32 + middle as f32 + 0.5,
                        edge as f32 + middle as f32 + 0.5,
                    ),
                    Vec3::X,
                ),
                (
                    Vec3::new(
                        -2.0,
                        edge as f32 + middle as f32 + 0.5,
                        edge as f32 + middle as f32 + 0.5,
                    ),
                    Vec3::X,
                ),
                (
                    Vec3::new(
                        -2.0,
                        edge as f32 + middle as f32 + 0.5,
                        edge as f32 + middle as f32 + 0.5,
                    ),
                    -Vec3::X,
                ),
            ];
            let probes: Vec<RayProbe> = ray_cases
                .iter()
                .map(|(origin, direction)| RayProbe {
                    origin: origin.extend(0.0).to_array(),
                    direction: direction.extend(0.0).to_array(),
                })
                .collect();
            let probe_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("GPU traversal test rays"),
                contents: bytemuck::cast_slice(&probes),
                usage: wgpu::BufferUsages::STORAGE,
            });
            let result_size = (probes.len() * std::mem::size_of::<[u32; 4]>()) as u64;
            let result_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("GPU traversal test results"),
                size: result_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("GPU traversal readback"),
                size: result_size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let raycast_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("GPU traversal test shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("raycast.wgsl").into()),
            });
            let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("GPU raycast test camera"),
                contents: bytemuck::bytes_of(&CameraUniform {
                    chunk_origin: [stream_origin.x, stream_origin.y, stream_origin.z, 0],
                    chunk_grid: [
                        stream_dimensions[0],
                        stream_dimensions[1],
                        stream_dimensions[2],
                        CHUNK_EDGE as u32,
                    ],
                    ..CameraUniform::zeroed()
                }),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let (output_texture, output_view) =
                create_raycast_output(&device, CHUNK_EDGE as u32, CHUNK_EDGE as u32);
            let image_bind_group_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("GPU image raycast test bind group layout"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::StorageTexture {
                                access: wgpu::StorageTextureAccess::WriteOnly,
                                format: RAYCAST_FORMAT,
                                view_dimension: wgpu::TextureViewDimension::D2,
                            },
                            count: None,
                        },
                    ],
                });
            let image_pipeline_layout =
                device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("GPU image raycast test pipeline layout"),
                    bind_group_layouts: &[Some(&image_bind_group_layout)],
                    immediate_size: 0,
                });
            let _image_pipeline =
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("GPU image raycast test pipeline"),
                    layout: Some(&image_pipeline_layout),
                    module: &raycast_shader,
                    entry_point: Some("raycast_image"),
                    compilation_options: Default::default(),
                    cache: None,
                });
            let _image_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("GPU image raycast test binding"),
                layout: &image_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: camera_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: voxel_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&output_view),
                    },
                ],
            });
            let _output_texture = output_texture;
            let mesh_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("GPU composition and debug test shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("mesh.wgsl").into()),
            });
            let render_bind_group_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("GPU mesh test bind group layout"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::VERTEX,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });
            let render_pipeline_layout =
                device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("GPU mesh test pipeline layout"),
                    bind_group_layouts: &[Some(&render_bind_group_layout)],
                    immediate_size: 0,
                });
            let _composite_pipeline =
                device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("GPU image composition test pipeline"),
                    layout: Some(&render_pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &mesh_shader,
                        entry_point: Some("composite_vs"),
                        compilation_options: Default::default(),
                        buffers: &[],
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &mesh_shader,
                        entry_point: Some("composite_fs"),
                        compilation_options: Default::default(),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: wgpu::TextureFormat::Rgba8Unorm,
                            blend: None,
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                    }),
                    primitive: wgpu::PrimitiveState::default(),
                    depth_stencil: None,
                    multisample: Default::default(),
                    multiview_mask: None,
                    cache: None,
                });
            let _debug_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("GPU chunk debug test pipeline"),
                layout: Some(&render_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &mesh_shader,
                    entry_point: Some("debug_vs"),
                    compilation_options: Default::default(),
                    buffers: &[Some(super::DebugVertex::layout())],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &mesh_shader,
                    entry_point: Some("debug_fs"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::LineList,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: Default::default(),
                multiview_mask: None,
                cache: None,
            });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("GPU traversal test pipeline"),
                layout: None,
                module: &raycast_shader,
                entry_point: Some("validate_rays"),
                compilation_options: Default::default(),
                cache: None,
            });
            let voxel_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("GPU traversal test voxel binding"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: camera_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: voxel_buffer.as_entire_binding(),
                    },
                ],
            });
            let probe_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("GPU traversal test ray binding"),
                layout: &pipeline.get_bind_group_layout(1),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: probe_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: result_buffer.as_entire_binding(),
                    },
                ],
            });
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("GPU traversal test encoder"),
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("GPU traversal test pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &voxel_bind_group, &[]);
                pass.set_bind_group(1, &probe_bind_group, &[]);
                pass.dispatch_workgroups(probes.len() as u32, 1, 1);
            }
            encoder.copy_buffer_to_buffer(&result_buffer, 0, &readback_buffer, 0, result_size);
            queue.submit([encoder.finish()]);

            let readback = readback_buffer.slice(..);
            let (sender, receiver) = mpsc::channel();
            readback.map_async(wgpu::MapMode::Read, move |result| {
                let _ = sender.send(result);
            });
            device
                .poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: None,
                })
                .unwrap();
            receiver.recv().unwrap().unwrap();
            let mapped = readback.get_mapped_range().unwrap();
            let gpu_results: &[[u32; 4]] = bytemuck::cast_slice(&mapped);

            for (index, ((origin, direction), gpu_hit)) in
                ray_cases.iter().zip(gpu_results).enumerate()
            {
                let cpu_hit = world.raycast(*origin, *direction, edge as f32 * 8.0);
                let expected = cpu_hit.map_or([u32::MAX, u32::MAX, u32::MAX, 0], |hit| {
                    [
                        hit.voxel[0] as u32,
                        hit.voxel[1] as u32,
                        hit.voxel[2] as u32,
                        hit.material as u32,
                    ]
                });
                assert_eq!(*gpu_hit, expected, "ray probe {index}");
            }
        });
    }
}
