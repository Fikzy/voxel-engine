use std::sync::Arc;
use std::time::Instant;

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

struct App {
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    startup_error: Option<anyhow::Error>,
    camera: Camera,
    camera_controller: CameraController,
    last_frame: Instant,
    mouse_look_active: bool,
}

impl Default for App {
    fn default() -> Self {
        Self {
            window: None,
            renderer: None,
            startup_error: None,
            camera: Camera::default(),
            camera_controller: CameraController::default(),
            last_frame: Instant::now(),
            mouse_look_active: false,
        }
    }
}

impl App {
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
                .with_inner_size(PhysicalSize::new(1280, 720)),
        ) {
            Ok(window) => Arc::new(window),
            Err(error) => {
                self.startup_error = Some(error.into());
                event_loop.exit();
                return;
            }
        };

        match pollster::block_on(Renderer::new(window.clone())) {
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
                if event.physical_key == PhysicalKey::Code(KeyCode::Escape)
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
                if let Some(renderer) = &mut self.renderer
                    && let Err(error) = renderer.render(
                        self.camera.view_projection(renderer.aspect_ratio()),
                        self.camera.position,
                    )
                {
                    self.startup_error = Some(error);
                    event_loop.exit();
                }
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
