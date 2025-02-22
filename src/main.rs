// MIT License

/* Copyright (c) 2024 Based Labs

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. */

mod api;
mod models;
mod systems;
mod server;
mod utils;

use serde_json::Value;
use std::sync::{Arc, Mutex};
use clap::{App, Arg};
use crate::models::types::Coordinates;
use crate::models::constants::{BATCH_SIZE, CELL_INIT_DELAY_MS, CYCLE_DELAY_MS};
use crate::systems::colony::Colony;
use rand::Rng;
use std::time::Duration;
use tokio::time;
use tokio::signal::ctrl_c;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc as StdArc;
use futures::future;
use serde_json::json;
use tokio::sync::mpsc::{self, Sender};

use crate::utils::animations::{AnimationStyle, AnimationConfig, ThinkingAnimation};

const DEFAULT_INITIAL_CELLS: usize = 32;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    crate::utils::logging::ensure_data_directories()
        .map_err(|e| Box::new(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())) as Box<dyn std::error::Error>)?;

    let running = StdArc::new(AtomicBool::new(true));
    let r = running.clone();
    let r2 = running.clone();

    let (shutdown_tx, _shutdown_rx) = tokio::sync::broadcast::channel(1);
    let shutdown_tx_clone = shutdown_tx.clone();

    let init_config = AnimationConfig {
        style: AnimationStyle::Progress,
        message: "Initializing Colony".to_string(),
        delay: Duration::from_millis(80),
    };
    let init_animation = ThinkingAnimation::new(init_config);

    tokio::spawn(async move {
        let mut sigterm = tokio::signal::unix::signal(
            tokio::signal::unix::SignalKind::terminate()
        ).expect("Failed to set up SIGTERM handler");
        
        tokio::select! {
            _ = ctrl_c() => {
                let shutdown_animation = ThinkingAnimation::new(AnimationConfig {
                    style: AnimationStyle::Progress,
                    message: "Shutting down...".to_string(),
                    delay: Duration::from_millis(100),
                });
                shutdown_animation.run().await.unwrap();
                
                for _ in 0..2 {
                    shutdown_tx_clone.send(()).unwrap();
                }
                println!("\nIf the process doesn't exit cleanly, force quit with:");
                println!("sudo kill -9 $(pgrep -fl 'creature' | awk '{{print $1}}')");
            }
            _ = sigterm.recv() => {
                let shutdown_animation = ThinkingAnimation::new(AnimationConfig {
                    style: AnimationStyle::Progress,
                    message: "Received SIGTERM".to_string(),
                    delay: Duration::from_millis(100),
                });
                shutdown_animation.run().await.unwrap();
                
                for _ in 0..2 {
                    if let Err(e) = shutdown_tx_clone.send(()) {
                        eprintln!("Failed to send shutdown signal: {}", e);
                    }
                }
            }
        }
        r.store(false, Ordering::SeqCst);
    });

    let matches = App::new("Creature")
        .version("0.1.0")
        .author("BasedAI")
        .about("Adaptive AI Colony Simulation")
        .arg(Arg::with_name("model")
            .short('l')
            .long("model")
            .value_name("MODEL")
            .help("Sets the Ollama model to use (default: llama3.1:8b)")
            .takes_value(true))
        .arg(Arg::with_name("mission")
            .short('m')
            .long("mission")
            .value_name("MISSION")
            .help("Sets the colony's mission")
            .takes_value(true))
        .arg(Arg::with_name("name")
            .short('n')
            .long("name")
            .value_name("NAME")
            .help("Sets the colony's name")
            .takes_value(true))
        .arg(Arg::with_name("state")
            .short('s')
            .long("state")
            .value_name("STATE_FILE")
            .help("Load initial state from file")
            .takes_value(true))
        .arg(Arg::with_name("cells")
            .short('c')
            .long("cells")
            .value_name("COUNT")
            .help("Sets the initial number of cells (default: 32)")
            .takes_value(true))
        .get_matches();

    let initial_cells = matches.value_of("cells")
        .and_then(|c| c.parse().ok())
        .unwrap_or(DEFAULT_INITIAL_CELLS);

    let mission = matches.value_of("mission")
        .unwrap_or("Develop innovative AI collaboration systems with focus on real-time adaptation")
        .to_string();
        
    let colony_name = matches.value_of("name").unwrap_or("Unnamed");
    
    let model = matches.value_of("model")
        .unwrap_or("llama3.1:8b")
        .to_string();

    let api_client = api::ollama::OllamaClient::new(model)
        .map_err(|e| e as Box<dyn std::error::Error>)?;
    let mut colony = Colony::new(&mission, api_client);

    let state_file = matches.value_of("state").unwrap_or("eca_state.json");
    if std::path::Path::new(state_file).exists() {
        let loading_animation = ThinkingAnimation::new(AnimationConfig {
            style: AnimationStyle::Progress,
            message: "Loading colony state".to_string(),
            delay: Duration::from_millis(50),
        });
        loading_animation.run().await?;
        
        match colony.load_state_from_file(state_file) {
            Ok(_) => println!("Loaded colony state from {}", state_file),
            Err(e) => eprintln!("Error loading state from {}: {}", state_file, e)
        }
    } else {
        if let Err(e) = colony.save_state_to_file("eca_state.json") {
            eprintln!("Error creating initial state file: {}", e);
        }
    }

    let colony = Arc::new(Mutex::new(colony));
    let colony_ws = Arc::clone(&colony);

    let shutdown_rx_ws = shutdown_tx.subscribe();
    tokio::spawn(async move {
        server::start_server(colony_ws, shutdown_rx_ws).await;
    });

    let simulation_cycles = 100000000;
    let mut current_cycle = 0;

    println!("Initializing colony...");
    init_animation.run().await?;

    let (init_tx, mut init_rx) = mpsc::channel::<Value>(100);
    let init_tx = Arc::new(init_tx);

    let init_progress = ThinkingAnimation::new(AnimationConfig {
        style: AnimationStyle::Progress,
        message: "Creating initial cells".to_string(),
        delay: Duration::from_millis(100),
    });

    let init_animation_handle = tokio::spawn(async move {
        init_progress.run().await
    });

    let mut init_futures = Vec::new();
    for cell_index in 0..initial_cells {
        if !running.load(Ordering::SeqCst) {
            break;
        }

        let colony = Arc::clone(&colony);
        let init_tx = Arc::clone(&init_tx);

        let future = tokio::spawn(async move {
            let grid_pos = (
                (cell_index as f64 / 9.0).floor(),
                ((cell_index % 9) as f64 / 3.0).floor(),
                (cell_index % 3) as f64
            );
            
            let x_offset = rand::thread_rng().gen_range(-0.2..0.2);
            let y_offset = rand::thread_rng().gen_range(-0.2..0.2);
            let z_offset = rand::thread_rng().gen_range(-0.2..0.2);
            
            let position = Coordinates {
                x: grid_pos.2 * 2.0 + x_offset,
                y: grid_pos.1 * 2.0 + y_offset,
                z: grid_pos.0 * 2.0 + z_offset,
                heat: 0.0,
                emergence_score: 0.0,
                coherence_score: 0.0,
                resilience_score: 0.0,
                intelligence_score: 0.0,
                efficiency_score: 0.0,
                integration_score: 0.0,
            };
            
            let (cell_id, cell) = {
                let mut colony_guard = colony.lock().unwrap();
                let id = colony_guard.add_cell(position.clone());
                let cell = colony_guard.cells.get(&id).unwrap().clone();
                (id, cell)
            };

            let init_event = json!({
                "type": "initialization",
                "message": format!(
                    "Initialized cell {} of {}:\n  Position: ({:.1}, {:.1}, {:.1})\n  Cell ID: {}\n  Energy: {:.1}",
                    cell_index + 1, initial_cells,
                    position.x, position.y, position.z,
                    cell_id,
                    cell.energy
                )
            });

            if let Err(e) = init_tx.send(init_event).await {
                eprintln!("Error sending initialization event: {}", e);
            }

            time::sleep(Duration::from_millis(CELL_INIT_DELAY_MS)).await;
            cell_id
        });

        init_futures.push(future);
    }

    let timeout_duration = Duration::from_secs(initial_cells as u64 * (CELL_INIT_DELAY_MS / 1000 + 1));
    match time::timeout(timeout_duration, future::join_all(init_futures)).await {
        Ok(results) => {
            for result in results {
                if let Err(e) = result {
                    eprintln!("Cell initialization error: {}", e);
                }
            }
        }
        Err(_) => {
            eprintln!("Colony initialization timed out!");
            return Err("Initialization timeout".into());
        }
    }

    if let Err(e) = init_animation_handle.await {
        eprintln!("Error in initialization animation: {}", e);
    }

    crate::utils::logging::print_banner(&mission, colony_name);
    'main: while current_cycle < simulation_cycles && running.load(Ordering::SeqCst) {
        let stats = {
            let colony_guard = colony.lock().unwrap();
            (
                colony_guard.cells.len(),
                colony_guard.get_average_energy(),
                colony_guard.get_total_thoughts(),
                colony_guard.get_total_plans(),
                colony_guard.get_mutation_rate(),
                colony_guard.get_cluster_count()
            )
        }; 

            
        println!("Active cells: {}", stats.0);
        println!("Average energy: {:.1}", stats.1);
        println!("Total thoughts: {}", stats.2);
        println!("Total plans: {}", stats.3);
        println!("Mutation rate: {:.1}%", stats.4 * 100.0);
        println!("Cluster count: {}", stats.5);
        
        let cell_ids: Vec<uuid::Uuid>;
        {
            let colony_guard = colony.lock().unwrap();
            cell_ids = colony_guard.cells.keys().copied().collect();
        }

        for batch_idx in (0..cell_ids.len()).step_by(BATCH_SIZE) {
            if !running.load(Ordering::SeqCst) {
                println!("Shutting down simulation...");
                break 'main;
            }

            let batch_animation = ThinkingAnimation::new(AnimationConfig {
                style: AnimationStyle::Spinner,
                message: format!("Processing batch {} of {}", 
                    batch_idx / BATCH_SIZE + 1, 
                    (cell_ids.len() + BATCH_SIZE - 1) / BATCH_SIZE),
                delay: Duration::from_millis(50),
            });
            batch_animation.run().await?;

            let batch_end = (batch_idx + BATCH_SIZE).min(cell_ids.len());
            let batch = cell_ids[batch_idx..batch_end].to_vec();
            if let Err(e) = colony.lock().unwrap().process_cell_batch(&batch).await {
                eprintln!("Error processing cell batch: {}", e);
            }
        }
        
        for batch_idx in (0..cell_ids.len()).step_by(BATCH_SIZE) {
            let batch_end = (batch_idx + BATCH_SIZE).min(cell_ids.len());
            let batch = cell_ids[batch_idx..batch_end].to_vec();
            
            println!("
╔════════════════════ PLAN GENERATION ═══════════════════╗");
            println!("║ Batch {}/{} - Processing {} cells", 
                batch_idx / BATCH_SIZE + 1, 
                (cell_ids.len() + BATCH_SIZE - 1) / BATCH_SIZE,
                batch.len()
            );
            println!("╠══════════════════════════════════════════════════════════╣");

            let plan_animation = ThinkingAnimation::new(AnimationConfig {
                style: AnimationStyle::Progress,
                message: format!("Creating plans for batch {}", batch_idx / BATCH_SIZE + 1),
                delay: Duration::from_millis(50),
            });
            plan_animation.run().await?;

            if let Err(e) = colony.lock().unwrap().create_plans_batch(&batch, &current_cycle.to_string()).await {
                eprintln!("Error creating plans batch: {}", e);
            }
                
            println!("╚══════════════════════════════════════════════════════════╝");
        }
        
        // Evolution phase
        let evolution_animation = ThinkingAnimation::new(AnimationConfig {
            style: AnimationStyle::Spinner,
            message: "Evolving cells".to_string(),
            delay: Duration::from_millis(80),
        });
        evolution_animation.run().await?;

        if let Err(e) = colony.lock().unwrap().evolve_cells().await {
            eprintln!("Error evolving cells: {}", e);
        }

        let reproduction_animation = ThinkingAnimation::new(AnimationConfig {
            style: AnimationStyle::Progress,
            message: "Cell reproduction in progress".to_string(),
            delay: Duration::from_millis(60),
        });
        reproduction_animation.run().await?;

        if let Err(e) = colony.lock().unwrap().handle_cell_reproduction().await {
            eprintln!("Error handling cell reproduction: {}", e);
        }

        let mission_animation = ThinkingAnimation::new(AnimationConfig {
            style: AnimationStyle::Spinner,
            message: "Updating mission progress".to_string(),
            delay: Duration::from_millis(70),
        });
        mission_animation.run().await?;

        if let Err(e) = colony.lock().unwrap().update_mission_progress().await {
            eprintln!("Error updating mission progress: {}", e);
        }
        
         // Memory compression (every other cycle)
         if current_cycle % 2 == 0 {
            let compression_animation = ThinkingAnimation::new(AnimationConfig {
                style: AnimationStyle::Progress,
                message: "Compressing colony memories".to_string(),
                delay: Duration::from_millis(50),
            });
            compression_animation.run().await?;

            if let Err(e) = colony.lock().unwrap().compress_colony_memories().await {
                eprintln!("Error compressing colony memories: {}", e);
            }
        }
        
        {
            let mut colony_guard = colony.lock().unwrap();
            colony_guard.print_cycle_statistics(current_cycle);
            if let Err(e) = colony_guard.save_state() {
                eprintln!("Error saving state: {}", e);
            }
            
            colony_guard.update_leaderboard();
            colony_guard.print_leaderboard();
        }
        
        current_cycle += 1;
        time::sleep(Duration::from_millis(CYCLE_DELAY_MS)).await;
        
        // Spawn thinking animation task
        let animation_running = running.clone();
        tokio::spawn(async move {
            let config = AnimationConfig {
                style: AnimationStyle::Spinner,
                message: "Thinking".to_string(),
                delay: Duration::from_millis(100),
            };
            let animation = ThinkingAnimation::new(config);
            let mut frame = 0;
            while animation_running.load(Ordering::SeqCst) {
                let _ = animation.update(frame).await;
                frame = (frame + 1) % 6;
            }
        });
    }

    if !running.load(Ordering::SeqCst) {
        println!("Shutting down gracefully...");
    } else {
        println!("Simulation complete!");
    }
    
    colony.lock().unwrap().print_statistics();
    
    let cleanup_animation = ThinkingAnimation::new(AnimationConfig {
        style: AnimationStyle::Progress,
        message: "Cleaning up resources".to_string(),
        delay: Duration::from_millis(100),
    });
    cleanup_animation.run().await?;
    
    // Resource cleanup
    drop(colony);
    let cleanup_timeout = Duration::from_secs(180);
    let cleanup_deadline = tokio::time::Instant::now() + cleanup_timeout;
    
    while tokio::time::Instant::now() < cleanup_deadline {
        if !running.load(Ordering::SeqCst) {
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    
    let completion_animation = ThinkingAnimation::new(AnimationConfig {
        style: AnimationStyle::Progress,
        message: "Shutdown complete".to_string(),
        delay: Duration::from_millis(150),
    });
    completion_animation.run().await?;

    println!("Shutdown complete");
    Ok(())
}