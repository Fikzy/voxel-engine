use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{DeviceEvent, ElementState, MouseButton, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{CursorGrabMode, Window, WindowId},
};

use crate::camera::Camera;
use crate::camera::controller::CameraController;
use crate::renderer::Renderer;
use crate::world::{STREAM_RADIUS, World};

struct App {
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    startup_error: Option<anyhow::Error>,
    camera: Camera,
    camera_controller: CameraController,
    world: World,
    debug_visualization: bool,
    last_frame: Instant,
    mouse_look_active: bool,
    timing_started: Instant,
    timing_frames: u32,
    stream_time: Duration,
    stream_time_max: Duration,
    sync_time: Duration,
    sync_time_max: Duration,
    render_time: Duration,
}

impl Default for App {
    fn default() -> Self {
        Self {
            window: None,
            renderer: None,
            startup_error: None,
            camera: Camera::default(),
            camera_controller: CameraController::default(),
            world: World::default(),
            debug_visualization: false,
            last_frame: Instant::now(),
            mouse_look_active: false,
            timing_started: Instant::now(),
            timing_frames: 0,
            stream_time: Duration::ZERO,
            stream_time_max: Duration::ZERO,
            sync_time: Duration::ZERO,
            sync_time_max: Duration::ZERO,
            render_time: Duration::ZERO,
        }
    }
}

impl App {
    fn report_frame_timings(
        &mut self,
        stream_time: Duration,
        sync_time: Duration,
        render_time: Duration,
    ) {
        self.timing_frames += 1;
        self.stream_time += stream_time;
        self.stream_time_max = self.stream_time_max.max(stream_time);
        self.sync_time += sync_time;
        self.sync_time_max = self.sync_time_max.max(sync_time);
        self.render_time += render_time;

        let elapsed = self.timing_started.elapsed();
        if elapsed < Duration::from_secs(1) {
            return;
        }

        let frames = self.timing_frames.max(1) as f64;
        let (generated_chunks, generation_time) = self.world.take_generation_timings();
        eprintln!(
            "[perf] fps={:.1} stream_poll_cpu_ms(avg/max)={:.3}/{:.3} chunk_sync_cpu_ms(avg/max)={:.3}/{:.3} render_submit_cpu_ms(avg)={:.3} generated={}/{}ms pending={} loaded={}",
            frames / elapsed.as_secs_f64(),
            self.stream_time.as_secs_f64() * 1000.0 / frames,
            self.stream_time_max.as_secs_f64() * 1000.0,
            self.sync_time.as_secs_f64() * 1000.0 / frames,
            self.sync_time_max.as_secs_f64() * 1000.0,
            self.render_time.as_secs_f64() * 1000.0 / frames,
            generated_chunks,
            generation_time.as_secs_f64() * 1000.0,
            self.world.pending_chunk_count(),
            self.world.loaded_chunk_count(),
        );
        self.timing_started = Instant::now();
        self.timing_frames = 0;
        self.stream_time = Duration::ZERO;
        self.stream_time_max = Duration::ZERO;
        self.sync_time = Duration::ZERO;
        self.sync_time_max = Duration::ZERO;
        self.render_time = Duration::ZERO;
    }

    fn set_mouse_look(&mut self, active: bool) {
        let Some(window) = &self.window else {
            return;
        };
        self.mouse_look_active = active;
        self.camera_controller.clear_input();
        if active {
            if window
                .set_cursor_grab(CursorGrabMode::Locked)
                .or_else(|_| window.set_cursor_grab(CursorGrabMode::Confined))
                .is_ok()
            {
                window.set_cursor_visible(false);
            } else {
                self.mouse_look_active = false;
            }
        } else {
            let _ = window.set_cursor_grab(CursorGrabMode::None);
            window.set_cursor_visible(true);
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let window = match event_loop.create_window(
            Window::default_attributes()
                .with_title("Voxel Engine")
                .with_inner_size(PhysicalSize::new(1920, 1080)),
        ) {
            Ok(window) => Arc::new(window),
            Err(error) => {
                self.startup_error = Some(error.into());
                event_loop.exit();
                return;
            }
        };

        self.world
            .stream_around(self.camera.position, STREAM_RADIUS);
        match pollster::block_on(Renderer::new(window.clone(), &self.world)) {
            Ok(renderer) => {
                self.renderer = Some(renderer);
                self.window = Some(window);
                self.last_frame = Instant::now();
            }
            Err(error) => {
                self.startup_error = Some(error);
                event_loop.exit();
            }
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: DeviceEvent,
    ) {
        if self.mouse_look_active
            && let DeviceEvent::MouseMotion { delta } = event
        {
            self.camera_controller.process_mouse_motion(delta);
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        if self
            .window
            .as_ref()
            .is_none_or(|window| window.id() != window_id)
        {
            return;
        }

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Focused(false) => self.set_mouse_look(false),
            WindowEvent::MouseInput {
                button: MouseButton::Left,
                state: ElementState::Pressed,
                ..
            } if !self.mouse_look_active => self.set_mouse_look(true),
            WindowEvent::KeyboardInput { event, .. } => {
                if event.physical_key == PhysicalKey::Code(KeyCode::F1)
                    && event.state.is_pressed()
                    && !event.repeat
                {
                    self.debug_visualization = !self.debug_visualization;
                    if let Some(renderer) = &mut self.renderer {
                        renderer.set_debug_visualization(self.debug_visualization);
                    }
                } else if event.physical_key == PhysicalKey::Code(KeyCode::Escape)
                    && event.state.is_pressed()
                    && !event.repeat
                {
                    if self.mouse_look_active {
                        self.set_mouse_look(false);
                    } else {
                        event_loop.exit();
                    }
                } else {
                    self.camera_controller.process_key(&event);
                }
            }
            WindowEvent::Resized(size) => {
                if let Some(renderer) = &mut self.renderer {
                    renderer.resize(size);
                }
            }
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                self.camera_controller
                    .update(&mut self.camera, now - self.last_frame);
                self.last_frame = now;
                let stream_started = Instant::now();
                let world_changed = self
                    .world
                    .stream_around(self.camera.position, STREAM_RADIUS);
                let stream_time = stream_started.elapsed();
                let mut sync_time = Duration::ZERO;
                let mut render_time = Duration::ZERO;
                if let Some(renderer) = &mut self.renderer {
                    if world_changed {
                        let changes = self.world.take_chunk_changes();
                        let sync_started = Instant::now();
                        renderer.update_world(&self.world, changes);
                        sync_time = sync_started.elapsed();
                    }
                    let render_started = Instant::now();
                    if let Err(error) = renderer.render(
                        self.camera.view_projection(renderer.aspect_ratio()),
                        self.camera.position,
                    ) {
                        self.startup_error = Some(error);
                        event_loop.exit();
                    }
                    render_time = render_started.elapsed();
                }
                self.report_frame_timings(stream_time, sync_time, render_time);
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}

pub fn run() -> Result<()> {
    let event_loop = EventLoop::new().context("failed to create event loop")?;
    let mut app = App::default();
    event_loop.run_app(&mut app).context("event loop failed")?;

    if let Some(error) = app.startup_error {
        return Err(error);
    }

    Ok(())
}
