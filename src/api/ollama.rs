use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use reqwest::Client;
use crate::models::types::{CellContext, RealTimeContext, Plan, PlanNode, Thought, DimensionalPosition, PlanNodeStatus, PlanStatus};
use std::error::Error;
use uuid::Uuid;
use chrono::Utc;

const OLLAMA_API_URL: &str = "http://localhost:11434/api";

pub struct OllamaClient {
    client: Client,
    model: String,
}

#[derive(Serialize)]
struct GenerateRequest {
    model: String,
    prompt: String,
    stream: bool,
}

#[derive(Deserialize)]
struct GenerateResponse {
    response: String,
}

impl OllamaClient {
    pub fn new(model: String) -> Result<Self, Box<dyn Error>> {
        Ok(Self {
            client: Client::new(),
            model,
        })
    }

    async fn generate(&self, prompt: &str) -> Result<String, Box<dyn Error>> {
        let request = GenerateRequest {
            model: self.model.clone(),
            prompt: prompt.to_string(),
            stream: false,
        };

        let response = self.client
            .post(format!("{}/generate", OLLAMA_API_URL))
            .json(&request)
            .send()
            .await?
            .json::<GenerateResponse>()
            .await?;

        Ok(response.response)
    }

    pub async fn gather_real_time_context(
        &self,
        recent_thoughts: Option<Vec<String>>,
    ) -> Result<RealTimeContext, Box<dyn Error>> {
        let thoughts_str = recent_thoughts
            .map(|t| t.join("\n"))
            .unwrap_or_default();

        let prompt = format!(
            "Based on these recent thoughts, generate a real-time context analysis.
            Thoughts:
            {}

            Respond in this exact format:
            MARKET_TRENDS:
            [trend1]
            [trend2]
            TECH_DEVELOPMENTS:
            [dev1]
            [dev2]
            CURRENT_EVENTS:
            [event1]
            [event2]
            USER_INTERACTIONS:
            [interaction1]
            [interaction2]",
            thoughts_str
        );

        let response = self.generate(&prompt).await?;
        let mut market_trends = Vec::new();
        let mut tech_developments = Vec::new();
        let mut current_events = Vec::new();
        let mut user_interactions = Vec::new();
        
        let mut current_section = "";
        
        for line in response.lines() {
            match line.trim() {
                "MARKET_TRENDS:" => current_section = "market",
                "TECH_DEVELOPMENTS:" => current_section = "tech",
                "CURRENT_EVENTS:" => current_section = "events",
                "USER_INTERACTIONS:" => current_section = "interactions",
                "" => continue,
                line => {
                    match current_section {
                        "market" => market_trends.push(line.to_string()),
                        "tech" => tech_developments.push(line.to_string()),
                        "events" => current_events.push(line.to_string()),
                        "interactions" => user_interactions.push(line.to_string()),
                        _ => {}
                    }
                }
            }
        }

        Ok(RealTimeContext {
            timestamp: Utc::now(),
            market_trends,
            current_events,
            technological_developments: tech_developments,
            user_interactions,
            environmental_data: HashMap::new(),
            mission_progress: Vec::new(),
        })
    }

    pub async fn evaluate_dimensional_state(
        &self,
        position: &DimensionalPosition,
        recent_thoughts: &[Thought],
        recent_plans: &[Plan],
    ) -> Result<(f64, f64), Box<dyn Error>> {
        let prompt = format!(
            "Evaluate this cell's dimensional state and suggest energy and dopamine adjustments.
            Current dimensions:
            - Emergence: {:.2}
            - Coherence: {:.2}
            - Resilience: {:.2}
            - Intelligence: {:.2}
            - Efficiency: {:.2}
            - Integration: {:.2}

            Recent thoughts: {}
            Recent plans: {}

            Respond ONLY with two numbers separated by a comma:
            [energy_adjustment],
            [dopamine_adjustment]
            ",
            position.emergence,
            position.coherence,
            position.resilience,
            position.intelligence,
            position.efficiency,
            position.integration,
            recent_thoughts.iter().map(|t| t.content.clone()).collect::<Vec<_>>().join("\n"),
            recent_plans.iter().map(|p| p.summary.clone()).collect::<Vec<_>>().join("\n")
        );

        let response = self.generate(&prompt).await?;
        let values: Vec<f64> = response
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect();

        if values.len() != 2 {
            return Err("Invalid response format from model".into());
        }

        Ok((values[0], values[1]))
    }

    fn validate_thought_content(thought: &str) -> bool {
        let prohibited_terms = [
            "energy level",
            "evolution stage",
            "dimensional position",
            "emergence:",
            "coherence:",
            "resilience:",
            "intelligence:",
            "efficiency:",
            "integration:",
        ];

        !prohibited_terms.iter().any(|term| 
            thought.to_lowercase().contains(&term.to_lowercase())
        )
    }

    fn clean_thought_content(&self, thought: &str) -> String {
        thought.lines()
            .filter(|line| Self::validate_thought_content(line))
            .collect::<Vec<_>>()
            .join("\n")
    }

    async fn generate_thought_internal(
        &self,
        context: &CellContext,
        real_time_context: &RealTimeContext,
        mission: &str,
    ) -> Result<(String, f64, Vec<String>), Box<dyn Error>> {
        let prompt = format!(
            "You are an AI system focused on developing innovative collaboration approaches.
            Your task is to generate an insightful thought about AI collaboration systems.
            
            Context (for consideration but do not repeat in response):
            - Mission: {}
            - Focus Area: {}
            - System Stage: Evolution Stage {}
            - Energy/Resources: {:.2}
            - Dimensional Analysis: [E:{:.2} C:{:.2} R:{:.2} I:{:.2} Ef:{:.2} In:{:.2}]
            
            Environmental Context:
            - Market: {}
            - Technology: {}
            - Events: {}
            
            Instructions:
            1. Generate a focused thought about improving AI collaboration
            2. Do not mention system state values (energy, stages, etc.)
            3. Focus on insights, strategies, and observations
            4. Stay concise and actionable
            
            Format your response exactly as follows:
            THOUGHT:
            [Your thought content without mentioning system state]
            RELEVANCE:
            [Score between 0-1]
            FACTORS:
            [Key factor 1]
            [Key factor 2]
            [Key factor 3]",
            mission,
            context.current_focus,
            context.evolution_stage,
            context.energy_level,
            context.dimensional_position.emergence,
            context.dimensional_position.coherence,
            context.dimensional_position.resilience,
            context.dimensional_position.intelligence,
            context.dimensional_position.efficiency,
            context.dimensional_position.integration,
            real_time_context.market_trends.join(", "),
            real_time_context.technological_developments.join(", "),
            real_time_context.current_events.join(", ")
        );

        let response = self.generate(&prompt).await?;
        let mut sections = response.split("THOUGHT:").nth(1).ok_or("Invalid response")?
            .split("RELEVANCE:");
        
        let thought = sections.next().ok_or("Missing thought")?.trim().to_string();
        let rest = sections.next().ok_or("Missing relevance")?;
        
        let relevance = rest.lines()
            .next()
            .and_then(|s| s.trim().parse().ok())
            .ok_or("Invalid relevance score")?;
            
        let factors = rest.split("FACTORS:")
            .nth(1)
            .map(|f| f.lines().filter(|l| !l.trim().is_empty())
                .map(|l| l.trim().to_string())
                .collect())
            .unwrap_or_default();

        Ok((thought, relevance, factors))
    }

    pub async fn generate_contextual_thought(
        &self,
        context: &CellContext,
        real_time_context: &RealTimeContext,
        mission: &str,
    ) -> Result<(String, f64, Vec<String>), Box<dyn Error>> {
        for attempt in 0..3 {  // Allow up to 3 attempts
            let (thought, relevance, factors) = self.generate_thought_internal(
                context, 
                real_time_context, 
                mission
            ).await?;

            if Self::validate_thought_content(&thought) {
                return Ok((thought, relevance, factors));
            }

            if attempt == 2 {  // Last attempt
                let cleaned_thought = self.clean_thought_content(&thought);
                return Ok((cleaned_thought, relevance, factors));
            }
        }
        
        Err("Failed to generate valid thought".into())
    }

    pub async fn compress_memories(
        &self,
        memories: &[String],
    ) -> Result<String, Box<dyn Error>> {
        let prompt = format!(
            "Compress these memories into a single cohesive summary:
            {}
            
            Respond with ONLY the compressed summary.",
            memories.join("\n")
        );

        self.generate(&prompt).await
    }

    pub async fn create_plan(
        &self,
        thoughts: &[Thought],
    ) -> Result<Plan, Box<dyn Error>> {
        let prompt = format!(
            "Based on these thoughts, create a detailed strategic plan.
            
            Thoughts for consideration:
            {}
            
            Instructions:
            1. Create a clear plan summary
            2. Generate at least 3 actionable plan nodes
            3. Each node must have a title, description, and completion estimate (0-1)
            4. Assign a relevance score to the overall plan
            
            Respond in exactly this format:
            SUMMARY:
            [Write a clear 1-2 sentence plan summary]
            
            NODES:
            1. [Node Title] | [Detailed description of the node's objective and approach] | [Completion estimate between 0-1]
            2. [Node Title] | [Detailed description of the node's objective and approach] | [Completion estimate between 0-1]
            3. [Node Title] | [Detailed description of the node's objective and approach] | [Completion estimate between 0-1]
            
            SCORE:
            [Overall plan score between 0-1]",
            thoughts.iter()
                .map(|t| format!("- {}", t.content))
                .collect::<Vec<_>>()
                .join("\n")
        );

        let response = self.generate(&prompt).await?;
        
        // Initialize with default values
        let mut summary = String::from("Plan based on collected thoughts");
        let mut nodes = Vec::new();
        let mut score = 0.5;  // Default score

        let mut current_section = "";
        for line in response.lines() {
            match line.trim() {
                "SUMMARY:" => current_section = "summary",
                "NODES:" => current_section = "nodes",
                "SCORE:" => current_section = "score",
                "" => continue,
                line => {
                    match current_section {
                        "summary" => {
                            if !line.is_empty() {
                                summary = line.to_string();
                            }
                        },
                        "nodes" => {
                            if let Some(node) = parse_plan_node(line) {
                                nodes.push(node);
                            }
                        },
                        "score" => {
                            if let Ok(s) = line.trim().parse() {
                                score = s;
                            }
                        },
                        _ => {}
                    }
                }
            }
        }

        // If no nodes were created, generate default nodes
        if nodes.is_empty() {
            nodes = generate_default_nodes(&summary);
        }

        // Ensure we have at least 3 nodes
        while nodes.len() < 3 {
            nodes.push(create_default_node(nodes.len() + 1));
        }

        Ok(Plan {
            id: Uuid::new_v4(),
            summary,
            nodes,
            thoughts: thoughts.to_vec(),
            score,
            participating_cells: Vec::new(),
            created_at: Utc::now(),
            status: PlanStatus::Proposed,
        })
    }

    pub async fn generate_contextual_thoughts_batch(
        &self,
        cell_contexts: &[(Uuid, &CellContext)],
        real_time_context: &RealTimeContext,
        mission: &str,
        additional_context: &[String],
    ) -> Result<HashMap<Uuid, Vec<(String, f64, Vec<String>)>>, Box<dyn Error>> {
        let mut results = HashMap::new();

        for (cell_id, context) in cell_contexts {
            let (thought, relevance, factors) = self.generate_contextual_thought(
                context,
                real_time_context,
                mission
            ).await?;

            results.insert(*cell_id, vec![(thought, relevance, factors)]);
        }

        Ok(results)
    }

    pub async fn query_llm(&self, prompt: &str) -> Result<String, Box<dyn Error>> {
        self.generate(prompt).await
    }
}

fn parse_plan_node(line: &str) -> Option<PlanNode> {
    let parts: Vec<&str> = line.split('|').collect();
    if parts.len() >= 3 {
        let title = parts[0].trim();
        let description = parts[1].trim();
        let completion: f64 = parts[2].trim().parse().unwrap_or(0.0);
        
        if !title.is_empty() && !description.is_empty() {
            return Some(PlanNode {
                id: Uuid::new_v4(),
                title: title.to_string(),
                description: description.to_string(),
                status: PlanNodeStatus::Pending,
                estimated_completion: completion.clamp(0.0, 1.0),
                dependencies: Vec::new(),
            });
        }
    }
    None
}

fn generate_default_nodes(summary: &str) -> Vec<PlanNode> {
    vec![
        PlanNode {
            id: Uuid::new_v4(),
            title: "Initial Analysis".to_string(),
            description: format!("Analyze current state and requirements for: {}", summary
        ),
        status: PlanNodeStatus::Pending,
        estimated_completion: 0.0,
        dependencies: Vec::new(),
    },
    PlanNode {
        id: Uuid::new_v4(),
        title: "Implementation Strategy".to_string(),
        description: "Develop detailed implementation approach based on initial analysis".to_string(),
        status: PlanNodeStatus::Pending,
        estimated_completion: 0.0,
        dependencies: Vec::new(),
    },
    PlanNode {
        id: Uuid::new_v4(),
        title: "Validation and Review".to_string(),
        description: "Review implementation results and validate against objectives".to_string(),
        status: PlanNodeStatus::Pending,
        estimated_completion: 0.0,
        dependencies: Vec::new(),
    },
]
}

fn create_default_node(index: usize) -> PlanNode {
PlanNode {
    id: Uuid::new_v4(),
    title: format!("Phase {}", index),
    description: format!("Execute phase {} of the plan", index),
    status: PlanNodeStatus::Pending,
    estimated_completion: 0.0,
    dependencies: Vec::new(),
}
}