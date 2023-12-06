pub mod camera;
pub mod camera_controller;
pub mod state;
pub mod storage_texture;
pub mod texture;

use circular_queue::CircularQueue;
use std::time::{Duration, Instant};
use winit::{
    dpi::LogicalSize,
    event::*,
    event_loop::*,
    keyboard::{KeyCode, PhysicalKey},
    window::WindowBuilder,
};

use crate::state::State;

const DT_QUEUE_SIZE: usize = 30;
const FPS_REFRESH_INTERVAL: Duration = Duration::from_secs(1);

pub async fn run() {
    let event_loop = EventLoop::new().unwrap();
    let window = WindowBuilder::new()
        .with_inner_size(LogicalSize::new(1920, 1080))
        .build(&event_loop)
        .unwrap();

    event_loop.set_control_flow(ControlFlow::Poll);

    let mut state = State::new(window).await;
    let mut last_render_time = Instant::now();

    let mut last_fps_print = Instant::now();
    let mut dt_queue = CircularQueue::with_capacity(DT_QUEUE_SIZE);

    let _ = event_loop.run(move |event, elwt| {
        match event {
            Event::AboutToWait => {
                state.window().request_redraw();
            }
            Event::DeviceEvent { event, .. } => match event {
                // Has to be here because WindowEvent::CursorMoved doesn't fire on MacOS
                DeviceEvent::MouseMotion { delta } => {
                    if state.mouse_grabbed {
                        state.camera_controller.process_mouse(delta.0, delta.1)
                    }
                }
                _ => {}
            },
            Event::WindowEvent { event, .. } => {
                if state.input(&event) {
                    return;
                }
                match event {
                    WindowEvent::Resized(physical_size) => {
                        state.resize(physical_size);
                    }
                    WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                        state.resize_scalar(scale_factor);
                    }
                    WindowEvent::RedrawRequested => {
                        let now = Instant::now();
                        let dt = now - last_render_time;
                        last_render_time = now;

                        dt_queue.push(dt.as_micros());
                        if last_fps_print.elapsed() > FPS_REFRESH_INTERVAL {
                            last_fps_print = Instant::now();
                            let fps = 1_000_000.0
                                / (dt_queue.iter().sum::<u128>() as f64 / DT_QUEUE_SIZE as f64);
                            state
                                .window
                                .set_title(&format!("Voxel Engine - {:.1} fps", fps));
                            // println!("{:.1} fps", fps);
                        }

                        state.update(dt);

                        match state.render() {
                            Ok(_) => {}
                            // Reconfigure the surface if lost
                            Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                            // The system is out of memory, we should probably quit
                            Err(wgpu::SurfaceError::OutOfMemory) => elwt.exit(),
                            // All other errors (Outdated, Timeout) should be resolved by the next frame
                            Err(e) => eprintln!("{:?}", e),
                        }
                    }
                    WindowEvent::KeyboardInput { event, .. } => match event.physical_key {
                        PhysicalKey::Code(key_code) => match key_code {
                            KeyCode::Escape => {
                                if !state.mouse_grabbed && event.state.is_pressed() && !event.repeat
                                {
                                    elwt.exit();
                                }
                            }
                            _ => {}
                        },
                        _ => {}
                    },
                    WindowEvent::CloseRequested => {
                        elwt.exit();
                    }
                    _ => {}
                }
            }
            _ => (),
        }
    });
}
