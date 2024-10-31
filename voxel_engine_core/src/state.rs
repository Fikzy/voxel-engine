use bytemuck;
use cgmath::num_traits::clamp;
use cgmath::{self, Vector3};
use noise::NoiseFn;
use std::io::empty;
use std::mem;
use std::time::Duration;
use wgpu::{self, util::DeviceExt};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::{
    event::WindowEvent,
    window::{CursorGrabMode, Window},
};

use crate::camera::{Camera, CameraUniform, Projection};
use crate::camera_controller::CameraController;
use crate::storage_texture::StorageTexture;
use crate::texture::Texture;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
}

impl Vertex {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![0 => Float32x3],
        }
    }
}

#[rustfmt::skip]
const WIRE_CUBE_VERTICES: &[Vertex] = &[
    Vertex { position: [0.0, 0.0, 0.0] },
    Vertex { position: [1.0, 0.0, 0.0] },
    Vertex { position: [1.0, 1.0, 0.0] },
    Vertex { position: [0.0, 1.0, 0.0] },
    Vertex { position: [0.0, 0.0, 1.0] },
    Vertex { position: [1.0, 0.0, 1.0] },
    Vertex { position: [1.0, 1.0, 1.0] },
    Vertex { position: [0.0, 1.0, 1.0] },
];

#[rustfmt::skip]
const WIRE_CUBE_INDICES: &[u16] = &[
    0, 1, 1, 2, 2, 3, 3, 0, // Front face
    4, 5, 5, 6, 6, 7, 7, 4, // Back face
    0, 4, 1, 5, 2, 6, 3, 7, // Side faces
];

#[derive(Copy, Clone)]
struct Instance {
    position: cgmath::Vector3<f32>,
    scale: f32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct InstanceRaw {
    model: [[f32; 4]; 4],
}

impl Instance {
    fn to_raw(&self) -> InstanceRaw {
        InstanceRaw {
            model: (cgmath::Matrix4::from_translation(self.position)
                * cgmath::Matrix4::from_scale(self.scale))
            .into(),
        }
    }
}

impl InstanceRaw {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<InstanceRaw>() as wgpu::BufferAddress,
            // We need to switch from using a step mode of Vertex to Instance
            // This means that our shaders will only change to use the next
            // instance when the shader starts processing a new instance
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                // A mat4 takes up 4 vertex slots as it is technically 4 vec4s. We need to define a slot
                // for each vec4. We'll have to reassemble the mat4 in the shader.
                wgpu::VertexAttribute {
                    offset: 0,
                    // While our vertex shader only uses locations 0, and 1 now, in later tutorials we'll
                    // be using 2, 3, and 4, for Vertex. We'll start at slot 5 not conflict with them later
                    shader_location: 5,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
                    shader_location: 6,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 8]>() as wgpu::BufferAddress,
                    shader_location: 7,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 12]>() as wgpu::BufferAddress,
                    shader_location: 8,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        }
    }
}

// 1 unit = 1 meter
// 16^3 voxels per unit
// 64^3 voxels per chunk
const CHUNK_SIZE: u32 = 64;
const MAX_DEPTH: u32 = 5;

// 16^3 chunks = 4096 chunks loaded around the player
// Number of chunks to load in each direction from the player
const CHUNK_LOAD_RADIUS: usize = 8;
const CHUNK_LOAD_RADIUS_ISIZE: isize = CHUNK_LOAD_RADIUS as isize;
const CHUNK_LOAD_RADIUS_VEC3_I32: cgmath::Vector3<i32> = cgmath::Vector3::new(
    CHUNK_LOAD_RADIUS as i32,
    CHUNK_LOAD_RADIUS as i32,
    CHUNK_LOAD_RADIUS as i32,
);
const CHUNK_LOAD_DIAMETER: usize = CHUNK_LOAD_RADIUS * 2;
const CHUNK_LOAD_DIAMETER_I32: i32 = CHUNK_LOAD_DIAMETER as i32;

pub struct State {
    pub surface: wgpu::Surface,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub config: wgpu::SurfaceConfiguration,
    pub size: winit::dpi::PhysicalSize<u32>,
    pub render_pipeline: wgpu::RenderPipeline,
    pub debug_pipeline: wgpu::RenderPipeline,
    pub compute_pipeline: wgpu::ComputePipeline,
    pub debug_depth_texture: Texture,
    pub gbuffer_storage_texture: StorageTexture,
    pub gbuffer_render_bind_group: wgpu::BindGroup,
    pub gbuffer_compute_bind_group: wgpu::BindGroup,
    pub camera: Camera,
    pub projection: Projection,
    pub camera_controller: CameraController,
    pub camera_uniform: CameraUniform,
    pub camera_buffer: wgpu::Buffer,
    pub camera_bind_group: wgpu::BindGroup,
    noise: noise::SuperSimplex,
    chunk_data: Vec<u32>,
    chunk_data_buffer: wgpu::Buffer,
    wire_cube_vertex_buffer: wgpu::Buffer,
    wire_cube_index_buffer: wgpu::Buffer,
    wire_cube_instances: Vec<InstanceRaw>,
    wire_cube_instance_buffer: wgpu::Buffer,
    player_chunk: cgmath::Vector3<i32>,
    pub debug: bool,
    pub debug_disable_chunk_loading: bool,
    pub mouse_pressed: bool,
    pub mouse_grabbed: bool,
    // The window must be declared after the surface so
    // it gets dropped after it as the surface contains
    // unsafe references to the window's resources.
    pub window: Window,
}

impl State {
    pub async fn new(window: Window) -> Self {
        let size = window.inner_size();

        // The instance is a handle to our GPU
        // Backends::all => Vulkan + Metal + DX12 + Browser WebGPU
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            flags: wgpu::InstanceFlags::default(),
            dx12_shader_compiler: Default::default(),
            gles_minor_version: wgpu::Gles3MinorVersion::default(),
        });

        // # Safety
        //
        // The surface needs to live as long as the window that created it.
        // State owns the window so this should be safe.
        let surface = unsafe { instance.create_surface(&window) }.unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    features: wgpu::Features::POLYGON_MODE_LINE
                        | wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES,
                    // WebGL doesn't support all of wgpu's features, so if
                    // we're building for the web we'll have to disable some.
                    limits: if cfg!(target_arch = "wasm32") {
                        wgpu::Limits::downlevel_webgl2_defaults()
                    } else {
                        wgpu::Limits::default()
                    },
                    label: None,
                },
                None, // Trace path
            )
            .await
            .unwrap();

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);
        println!("Surface caps: {:?}", surface_caps.formats);
        println!("Surface format: {:?}", surface_format);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_DST,
            format: surface_format, // Rgba8UnormSrgb
            width: size.width,
            height: size.height,
            // present_mode: surface_caps.present_modes[0],
            present_mode: wgpu::PresentMode::AutoVsync,
            // present_mode: wgpu::PresentMode::Immediate, // Vsync off
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &config);

        let render_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Render Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("render.wgsl").into()),
        });

        let debug_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Debug Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("debug.wgsl").into()),
        });

        let compute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Compute"),
            source: wgpu::ShaderSource::Wgsl(include_str!("compute.wgsl").into()),
        });

        let debug_depth_texture =
            Texture::create_depth_texture(&device, &config, "Debug Depth Texture");

        let gbuffer_storage_texture =
            StorageTexture::create_storage_texture(&device, &config, "GBuffer Storage Texture");

        let chunk_data = vec![0; CHUNK_LOAD_DIAMETER.pow(3)];

        let chunk_data_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Chunk Data Buffer"),
            contents: bytemuck::cast_slice(&chunk_data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let camera = Camera::new((0.0, 0.0, 64.0), cgmath::Deg(-90.0), cgmath::Deg(0.0));
        let projection =
            Projection::new(config.width, config.height, cgmath::Deg(45.0), 0.1, 100.0);
        let camera_controller = CameraController::new(10.0, 20.0, 0.2);

        let mut camera_uniform = CameraUniform::new();
        camera_uniform.update_view_proj(&camera, &projection);

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Camera Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
            label: Some("Camera Bind Group"),
        });

        let gbuffer_render_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("GBuffer Render Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::ReadOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                }],
            });

        let gbuffer_render_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("GBuffer Render Bind Group"),
            layout: &gbuffer_render_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&gbuffer_storage_texture.view),
            }],
        });

        let gbuffer_compute_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("GBuffer Compute Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rgba8Unorm,
                            view_dimension: wgpu::TextureViewDimension::D2,
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
                ],
            });

        let gbuffer_compute_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("GBuffer Compute Bind Group"),
            layout: &gbuffer_compute_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&gbuffer_storage_texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(
                        chunk_data_buffer.as_entire_buffer_binding(),
                    ),
                },
            ],
        });

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&gbuffer_render_bind_group_layout],
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &render_shader,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &render_shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        let debug_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Debug Pipeline Layout"),
                bind_group_layouts: &[&camera_bind_group_layout],
                push_constant_ranges: &[],
            });

        let debug_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Debug Pipeline"),
            layout: Some(&debug_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &debug_shader,
                entry_point: "vs_main",
                buffers: &[Vertex::desc(), InstanceRaw::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &debug_shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Line,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: Texture::DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        let compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Compute Pipeline Layout"),
                bind_group_layouts: &[
                    &camera_bind_group_layout,
                    &gbuffer_compute_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &compute_shader,
            entry_point: "main",
        });

        let wire_cube_vertex_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Wire Cube Vertex Buffer"),
                contents: bytemuck::cast_slice(WIRE_CUBE_VERTICES),
                usage: wgpu::BufferUsages::VERTEX,
            });

        let wire_cube_index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Wire Cube Index Buffer"),
            contents: bytemuck::cast_slice(WIRE_CUBE_INDICES),
            usage: wgpu::BufferUsages::INDEX,
        });

        let mut wire_cube_instances = vec![
            InstanceRaw {
                model: Default::default(),
            };
            CHUNK_LOAD_DIAMETER.pow(3)
        ];

        let wire_cube_instance_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Wire Cube Instance Buffer"),
                contents: bytemuck::cast_slice(&wire_cube_instances),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            });

        Self {
            window,
            surface,
            device,
            queue,
            config,
            size,
            render_pipeline,
            debug_pipeline,
            compute_pipeline,
            debug_depth_texture,
            gbuffer_storage_texture,
            gbuffer_render_bind_group,
            gbuffer_compute_bind_group,
            camera,
            projection,
            camera_controller,
            camera_uniform,
            camera_buffer,
            camera_bind_group,
            noise: noise::SuperSimplex::new(0),
            chunk_data,
            chunk_data_buffer,
            wire_cube_vertex_buffer,
            wire_cube_index_buffer,
            wire_cube_instances,
            wire_cube_instance_buffer,
            player_chunk: cgmath::Vector3::new(0, 0, 0),
            debug: false,
            debug_disable_chunk_loading: false,
            mouse_pressed: false,
            mouse_grabbed: false,
        }
    }

    pub fn window(&self) -> &Window {
        &self.window
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
            self.gbuffer_storage_texture = StorageTexture::create_storage_texture(
                &self.device,
                &self.config,
                "GBuffer Storage Texture",
            );
            self.gbuffer_render_bind_group =
                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("GBuffer Render Bind Group"),
                    layout: &self.render_pipeline.get_bind_group_layout(0),
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(
                            &self.gbuffer_storage_texture.view,
                        ),
                    }],
                });
            self.gbuffer_compute_bind_group =
                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("GBuffer Compute Bind Group"),
                    layout: &self.compute_pipeline.get_bind_group_layout(1),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(
                                &self.gbuffer_storage_texture.view,
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Buffer(
                                self.chunk_data_buffer.as_entire_buffer_binding(),
                            ),
                        },
                    ],
                });
            self.debug_depth_texture =
                Texture::create_depth_texture(&self.device, &self.config, "Debug Depth Texture");
            self.projection.resize(new_size.width, new_size.height);
        }
    }

    pub fn resize_scalar(&mut self, scalar: f64) {
        let new_size = winit::dpi::PhysicalSize::new(
            (self.size.width as f64 * scalar) as u32,
            (self.size.height as f64 * scalar) as u32,
        );
        self.resize(new_size);
    }

    pub fn input(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::KeyboardInput { event, .. } => {
                match event.physical_key {
                    PhysicalKey::Code(key_code) => match key_code {
                        KeyCode::F1 => {
                            if event.state.is_pressed() {
                                self.debug = !self.debug;
                                return true;
                            }
                        }
                        KeyCode::Escape => {
                            if self.mouse_grabbed && event.state.is_pressed() {
                                self.mouse_pressed = false;
                                self.mouse_grabbed = false;
                                self.window.set_cursor_grab(CursorGrabMode::None).unwrap();
                                self.window.set_cursor_visible(true);
                                return true;
                            } else {
                                return false;
                            }
                        }
                        _ => {}
                    },
                    _ => {}
                }
                self.camera_controller.process_keyboard_event(event)
            }
            WindowEvent::MouseInput { state, .. } => {
                if state.is_pressed() {
                    self.mouse_pressed = true;
                    self.mouse_grabbed = true;
                    self.window
                        .set_cursor_grab(CursorGrabMode::Confined)
                        .unwrap();
                    self.window.set_cursor_visible(false);
                }
                true
            }
            WindowEvent::MouseWheel { delta, .. } => {
                self.camera_controller.process_scroll(delta);
                true
            }
            _ => false,
        }
    }

    /// Generate a chunk of voxels
    /// * `depth` - The current depth in the octree
    /// * `octant_indices` - The indices of the octant in the parent octant
    pub fn generate_chunk(&mut self, depth: u32, mut octant_indices: Vector3<u32>) -> Vec<u32> {
        let mut chunk_data = Vec::<u32>::new();

        if depth == 0 {
            for idx in 0..8 {
                octant_indices.x |= (idx & 0b100 >> 2) << depth;
                octant_indices.y |= (idx & 0b010 >> 1) << depth;
                octant_indices.z |= (idx & 0b001 >> 0) << depth;
                self.generate_chunk(depth + 1, octant_indices);
            }

            let child_descriptor = 0u32;
            chunk_data.push(0); // First chunk descriptor 
        }

        if depth == MAX_DEPTH {
            let mut octant_coord = Vector3::new(0, 0, 0);

            for d in 0..depth {
                octant_coord.x += ((octant_indices.x & 1 << d) >> d) * (CHUNK_SIZE >> (d + 1));
                octant_coord.y += ((octant_indices.y & 1 << d) >> d) * (CHUNK_SIZE >> (d + 1));
                octant_coord.z += ((octant_indices.z & 1 << d) >> d) * (CHUNK_SIZE >> (d + 1));
            }
            println!(
                "Creating leaf voxel at: {}, {}, {}",
                octant_coord.x, octant_coord.y, octant_coord.z
            );

            let mut valid_mask = 0u32;
            let mut leaf_mask = 0u32;
            for idx in 0..8 {
                let voxel_coord = cgmath::Vector3::new(
                    octant_coord.x + (idx & 0b100 >> 2),
                    octant_coord.y + (idx & 0b010 >> 1),
                    octant_coord.z + (idx & 0b001 >> 0),
                );
                let voxel = self.noise.get([
                    voxel_coord.x as f64 / CHUNK_SIZE as f64,
                    voxel_coord.y as f64 / CHUNK_SIZE as f64,
                    voxel_coord.z as f64 / CHUNK_SIZE as f64,
                ]);
                if voxel > 0.0 {
                    valid_mask |= 1 << idx;
                    leaf_mask |= 1 << idx;
                }
            }

            let mut child_descriptor = 0u32;
            child_descriptor |= valid_mask << 8;
            child_descriptor |= leaf_mask;

            return vec![child_descriptor];
        }

        for x in 0..CHUNK_SIZE {
            for y in 0..CHUNK_SIZE {
                for z in 0..CHUNK_SIZE {
                    // let voxel_coord = cgmath::Vector3::new(
                    //     current_player_chunk.x * CHUNK_SIZE as i32 + x as i32,
                    //     current_player_chunk.y * CHUNK_SIZE as i32 + y as i32,
                    //     current_player_chunk.z * CHUNK_SIZE as i32 + z as i32,
                    // );

                    // let voxel = self.noise.get([
                    //     voxel_coord.x as f64 / CHUNK_SIZE as f64,
                    //     voxel_coord.y as f64 / CHUNK_SIZE as f64,
                    //     voxel_coord.z as f64 / CHUNK_SIZE as f64,
                    // ]);

                    let voxel = self.noise.get([
                        x as f64 / CHUNK_SIZE as f64,
                        y as f64 / CHUNK_SIZE as f64,
                        z as f64 / CHUNK_SIZE as f64,
                    ]);

                    if voxel > 0.0 {
                        let mut idx = 0;
                        let mut depth = 1;
                        let mut octant_coord: Vector3<u32> = Vector3::new(0, 0, 0);

                        loop {
                            let mut child_descriptor = chunk_data[idx];
                            let mut child_pointer = child_descriptor & 0xfffe0000 >> 17;
                            let mut valid_mask = child_descriptor & 0xff00 >> 8;
                            let mut leaf_mask = child_descriptor & 0xff;

                            // FIXME: children must be stored contiguously

                            if depth == MAX_DEPTH {
                                // Set leaf mask
                                println!("Creating leaf voxel at: {}, {}, {}", x, y, z);

                                octant_coord.x =
                                    (x - octant_coord.x * CHUNK_SIZE) / (CHUNK_SIZE >> depth);
                                octant_coord.y =
                                    (y - octant_coord.y * CHUNK_SIZE) / (CHUNK_SIZE >> depth);
                                octant_coord.z =
                                    (z - octant_coord.z * CHUNK_SIZE) / (CHUNK_SIZE >> depth);
                                let octant_idx =
                                    octant_coord.x << 2 | octant_coord.y << 1 | octant_coord.z;

                                valid_mask |= 1 << octant_idx;
                                leaf_mask |= 1 << octant_idx;

                                println!("Valid mask: {:b}", valid_mask);
                                println!("Leaf mask: {:b}", leaf_mask);

                                child_descriptor |= valid_mask << 8;
                                child_descriptor |= leaf_mask;

                                chunk_data[idx] = child_descriptor; // Update parent descriptor

                                break;
                            }

                            if child_pointer == 0 {
                                // Create new children
                                println!("Creating new children for voxel: {}, {}, {}", x, y, z);
                                println!("Idx: {:?}", idx);
                                println!("Depth: {:?}", depth);
                                println!("Octant coord: {:?}", octant_coord);

                                octant_coord.x =
                                    (x - octant_coord.x * CHUNK_SIZE) / (CHUNK_SIZE >> depth);
                                octant_coord.y =
                                    (y - octant_coord.y * CHUNK_SIZE) / (CHUNK_SIZE >> depth);
                                octant_coord.z =
                                    (z - octant_coord.z * CHUNK_SIZE) / (CHUNK_SIZE >> depth);
                                let octant_idx =
                                    octant_coord.x << 2 | octant_coord.y << 1 | octant_coord.z;

                                chunk_data.push(0); // New chunk descriptor

                                child_pointer = chunk_data.len() as u32;
                                valid_mask |= 1 << octant_idx;

                                println!("Valid mask: {:b}", valid_mask);

                                child_descriptor |= child_pointer << 17;
                                child_descriptor |= valid_mask << 8;
                                chunk_data[idx] = child_descriptor; // Update parent descriptor

                                idx = child_pointer as usize;
                                depth += 1;

                                continue;
                            } else {
                                // Traverse children
                                println!("Traversing children for voxel: {}, {}, {}", x, y, z);
                                idx = child_pointer as usize;
                                depth += 1; // FIXME: wrong
                            }
                        }

                        println!();
                    }
                }
            }
        }
        chunk_data
    }

    pub fn update(&mut self, dt: Duration) {
        self.camera_controller.update_camera(&mut self.camera, dt);
        self.camera_uniform
            .update_view_proj(&self.camera, &self.projection);

        let current_player_chunk = cgmath::Vector3::new(
            (self.camera.position.x / CHUNK_SIZE as f32).floor() as i32,
            (self.camera.position.y / CHUNK_SIZE as f32).floor() as i32,
            (self.camera.position.z / CHUNK_SIZE as f32).floor() as i32,
        );
        if current_player_chunk != self.player_chunk && !self.debug_disable_chunk_loading {
            self.debug_disable_chunk_loading = true;

            println!("Player moved to chunk: {:?}", current_player_chunk);

            let new_chunk_data = self.generate_chunk(0, Vector3::new(0, 0, 0));
            // Copy new chunk data to chunk data buffer

            self.player_chunk = current_player_chunk;

            self.queue.write_buffer(
                &self.chunk_data_buffer,
                0,
                bytemuck::cast_slice(&self.chunk_data),
            );

            self.queue.write_buffer(
                &self.wire_cube_instance_buffer,
                0,
                bytemuck::cast_slice(&self.wire_cube_instances),
            );
        }

        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[self.camera_uniform]),
        );
    }

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;

        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&self.compute_pipeline);

            compute_pass.set_bind_group(0, &self.camera_bind_group, &[]);
            compute_pass.set_bind_group(1, &self.gbuffer_compute_bind_group, &[]);

            compute_pass.dispatch_workgroups(self.config.width, self.config.height, 1);
        }

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            render_pass.set_pipeline(&self.render_pipeline);

            render_pass.set_bind_group(0, &self.gbuffer_render_bind_group, &[]);

            render_pass.draw(0..3, 0..1);
        }

        if self.debug {
            let mut debug_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Debug Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.debug_depth_texture.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            debug_pass.set_pipeline(&self.debug_pipeline);

            debug_pass.set_bind_group(0, &self.camera_bind_group, &[]);

            debug_pass.set_vertex_buffer(0, self.wire_cube_vertex_buffer.slice(..));
            debug_pass.set_vertex_buffer(1, self.wire_cube_instance_buffer.slice(..));
            debug_pass.set_index_buffer(
                self.wire_cube_index_buffer.slice(..),
                wgpu::IndexFormat::Uint16,
            );

            debug_pass.draw_indexed(
                0..WIRE_CUBE_INDICES.len() as _,
                0,
                0..self.wire_cube_instances.len() as _,
            );
        }

        self.queue.submit([encoder.finish()]);
        output.present();

        Ok(())
    }
}
