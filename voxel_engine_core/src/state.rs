use bytemuck;
use cgmath;
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
const FRONT_FACE_VERTICES: &[Vertex] = &[
    Vertex { position: [-0.5,  0.5, 0.5] },
    Vertex { position: [-0.5, -0.5, 0.5] },
    Vertex { position: [ 0.5, -0.5, 0.5] },
    Vertex { position: [ 0.5,  0.5, 0.5] },
];
#[rustfmt::skip]
const BACK_FACE_VERTICES: &[Vertex] = &[
    Vertex { position: [ 0.5,  0.5, -0.5] },
    Vertex { position: [ 0.5, -0.5, -0.5] },
    Vertex { position: [-0.5, -0.5, -0.5] },
    Vertex { position: [-0.5,  0.5, -0.5] },
];
#[rustfmt::skip]
const LEFT_FACE_VERTICES: &[Vertex] = &[
    Vertex { position: [-0.5,  0.5, -0.5] },
    Vertex { position: [-0.5, -0.5, -0.5] },
    Vertex { position: [-0.5, -0.5,  0.5] },
    Vertex { position: [-0.5,  0.5,  0.5] },
];
#[rustfmt::skip]
const RIGHT_FACE_VERTICES: &[Vertex] = &[
    Vertex { position: [ 0.5,  0.5,  0.5] },
    Vertex { position: [ 0.5, -0.5,  0.5] },
    Vertex { position: [ 0.5, -0.5, -0.5] },
    Vertex { position: [ 0.5,  0.5, -0.5] },
];
#[rustfmt::skip]
const BOTTOM_FACE_VERTICES: &[Vertex] = &[
    Vertex { position: [-0.5, -0.5,  0.5] },
    Vertex { position: [-0.5, -0.5, -0.5] },
    Vertex { position: [ 0.5, -0.5, -0.5] },
    Vertex { position: [ 0.5, -0.5,  0.5] },
];
#[rustfmt::skip]
const TOP_FACE_VERTICES: &[Vertex] = &[
    Vertex { position: [-0.5,  0.5, -0.5] },
    Vertex { position: [-0.5,  0.5,  0.5] },
    Vertex { position: [ 0.5,  0.5,  0.5] },
    Vertex { position: [ 0.5,  0.5, -0.5] },
];

const FACE_VERTICES: &[&[Vertex]] = &[
    FRONT_FACE_VERTICES,
    BACK_FACE_VERTICES,
    LEFT_FACE_VERTICES,
    RIGHT_FACE_VERTICES,
    BOTTOM_FACE_VERTICES,
    TOP_FACE_VERTICES,
];

const FACE_INDICES: &[u16] = &[0, 1, 3, 1, 2, 3];

struct Instance {
    position: cgmath::Vector3<f32>,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct InstanceRaw {
    model: [[f32; 4]; 4],
}

impl Instance {
    fn to_raw(&self) -> InstanceRaw {
        InstanceRaw {
            model: cgmath::Matrix4::from_translation(self.position).into(),
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

const CHUNK_SIZE: usize = 16;

const CHUNK_DISPLACEMENT: cgmath::Vector3<f32> = cgmath::Vector3::new(
    CHUNK_SIZE as f32 * 0.5,
    CHUNK_SIZE as f32 * 0.5,
    CHUNK_SIZE as f32 * 0.5,
);

pub struct State {
    pub surface: wgpu::Surface,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub config: wgpu::SurfaceConfiguration,
    pub size: winit::dpi::PhysicalSize<u32>,
    pub render_pipeline: wgpu::RenderPipeline,
    pub depth_texture: Texture,
    pub camera: Camera,
    pub projection: Projection,
    pub camera_controller: CameraController,
    pub camera_uniform: CameraUniform,
    pub camera_buffer: wgpu::Buffer,
    pub camera_bind_group: wgpu::BindGroup,
    face_vertex_buffers: [wgpu::Buffer; 6],
    face_index_buffer: wgpu::Buffer,
    // chunk_data: [[[u8; CHUNK_SIZE]; CHUNK_SIZE]; CHUNK_SIZE],
    face_instances: [Vec<InstanceRaw>; 6],
    face_instance_buffers: [wgpu::Buffer; 6],
    // The window must be declared after the surface so
    // it gets dropped after it as the surface contains
    // unsafe references to the window's resources.
    pub window: Window,
    pub mouse_pressed: bool,
    pub mouse_grabbed: bool,
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
                    features: wgpu::Features::POLYGON_MODE_LINE,
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
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            // present_mode: surface_caps.present_modes[0],
            present_mode: wgpu::PresentMode::Immediate, // Vsync off
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &config);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let depth_texture = Texture::create_depth_texture(&device, &config, "depth_texture");

        let camera = Camera::new((0.0, 5.0, 10.0), cgmath::Deg(-90.0), cgmath::Deg(-20.0));
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
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("camera_bind_group_layout"),
            });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
            label: Some("camera_bind_group"),
        });

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&camera_bind_group_layout],
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[Vertex::desc(), InstanceRaw::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
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
                cull_mode: Some(wgpu::Face::Back),
                // polygon_mode: wgpu::PolygonMode::Fill,
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

        let face_vertex_buffers = FACE_VERTICES
            .iter()
            .map(|face_vertices| {
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Face Vertex Buffer"),
                    contents: bytemuck::cast_slice(face_vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                })
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        let face_index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(FACE_INDICES),
            usage: wgpu::BufferUsages::INDEX,
        });

        const CHUNK_CENTER: cgmath::Vector3<f32> = cgmath::Vector3::new(
            CHUNK_SIZE as f32 * 0.5,
            CHUNK_SIZE as f32 * 0.5,
            CHUNK_SIZE as f32 * 0.5,
        );
        const CHUNK_RADIUS_SQUARED: f32 = (CHUNK_SIZE / 2 * CHUNK_SIZE / 2) as f32;

        let mut chunk_data = [[[0; CHUNK_SIZE]; CHUNK_SIZE]; CHUNK_SIZE];
        for x in 0..CHUNK_SIZE {
            for y in 0..CHUNK_SIZE {
                for z in 0..CHUNK_SIZE {
                    // Sphere
                    let dist_squared = (x as f32 - CHUNK_CENTER.x).powi(2)
                        + (y as f32 - CHUNK_CENTER.y).powi(2)
                        + (z as f32 - CHUNK_CENTER.z).powi(2);
                    if dist_squared < CHUNK_RADIUS_SQUARED {
                        chunk_data[x][y][z] = 1 as u8;
                    }
                }
            }
        }

        let mut face_instances: [Vec<InstanceRaw>; 6] = Default::default();
        for x in 0..CHUNK_SIZE {
            for y in 0..CHUNK_SIZE {
                for z in 0..CHUNK_SIZE {
                    if chunk_data[x][y][z] == 0 {
                        continue;
                    }

                    let instance = Instance {
                        position: cgmath::Vector3 {
                            x: x as f32,
                            y: y as f32,
                            z: z as f32,
                        } - CHUNK_DISPLACEMENT,
                    }
                    .to_raw();

                    if z == CHUNK_SIZE - 1 || chunk_data[x][y][z + 1] == 0 {
                        face_instances[0].push(instance);
                    }
                    if z == 0 || chunk_data[x][y][z - 1] == 0 {
                        face_instances[1].push(instance);
                    }
                    if x == 0 || chunk_data[x - 1][y][z] == 0 {
                        face_instances[2].push(instance);
                    }
                    if x == CHUNK_SIZE - 1 || chunk_data[x + 1][y][z] == 0 {
                        face_instances[3].push(instance);
                    }
                    if y == 0 || chunk_data[x][y - 1][z] == 0 {
                        face_instances[4].push(instance);
                    }
                    if y == CHUNK_SIZE - 1 || chunk_data[x][y + 1][z] == 0 {
                        face_instances[5].push(instance);
                    }
                }
            }
        }

        let face_instance_buffers: [wgpu::Buffer; 6] = face_instances
            .iter()
            .map(|face_instances| {
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Face Instance Buffer"),
                    contents: bytemuck::cast_slice(face_instances),
                    usage: wgpu::BufferUsages::VERTEX,
                })
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        Self {
            window,
            surface,
            device,
            queue,
            config,
            size,
            render_pipeline,
            depth_texture,
            camera,
            projection,
            camera_controller,
            camera_uniform,
            camera_buffer,
            camera_bind_group,
            face_vertex_buffers,
            face_index_buffer,
            // chunk_data,
            face_instances,
            face_instance_buffers,
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
            self.depth_texture =
                Texture::create_depth_texture(&self.device, &self.config, "depth_texture");
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

    pub fn update(&mut self, dt: Duration) {
        self.camera_controller.update_camera(&mut self.camera, dt);
        self.camera_uniform
            .update_view_proj(&self.camera, &self.projection);
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
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            render_pass.set_pipeline(&self.render_pipeline);

            render_pass.set_bind_group(0, &self.camera_bind_group, &[]);

            for i in 0..6 {
                render_pass.set_vertex_buffer(0, self.face_vertex_buffers[i].slice(..));
                render_pass.set_vertex_buffer(1, self.face_instance_buffers[i].slice(..));
                render_pass
                    .set_index_buffer(self.face_index_buffer.slice(..), wgpu::IndexFormat::Uint16);

                render_pass.draw_indexed(
                    0..FACE_INDICES.len() as _,
                    0,
                    0..self.face_instances[i].len() as _,
                );
            }
        }

        self.queue.submit([encoder.finish()]);
        output.present();

        Ok(())
    }
}
