use rayon::prelude::*;
use std::sync::{Arc, Mutex};
use openai_api::prelude::*;
use serde::{Serialize, Deserialize};

#[derive(Debug)]
pub struct NodeError {
    pub msg: String
}

impl NodeError {
    pub fn new(msg: String) -> NodeError {
        NodeError { msg }
    }
    pub fn from(msg: &str) -> NodeError {
        NodeError { msg: msg.to_owned() }
    }
}


#[derive(Debug, Clone)]
pub struct MutexWrapper<T: ?Sized>(
    pub Arc<Mutex<T>>
);

mod mutexwrapper_serde {
    use serde::{Deserialize, Serialize};
    use super::{MutexWrapper, Node};

    pub fn serialize<S>(val: &Option<MutexWrapper<Node>>, s: S) -> Result<S::Ok, S::Error>
        where S: serde::Serializer,
              Node: Serialize,
    {
        match &val.as_ref() {
            Some(val) => { 
                let test = val.0.clone();
                let test = test.lock().expect("Failed to lock mutex for serialization");
                Option::<Node>::serialize(&Some(test.clone()),s)},
            None => {Option::<Node>::serialize(&Option::<Node>::default(),s)}
        }
    }

    pub fn deserialize<'de, D, T>(d: D) -> Result<Option<MutexWrapper<T>>, D::Error>
        where D: serde::Deserializer<'de>,
              T: Deserialize<'de>,
    {
        let node = Option::<T>::deserialize(d)?;
        match node {
            Some(node) => {Ok(Some(MutexWrapper::<T>::new(node)))},
            None => {Ok(None)}
        }
    }
}

impl<T> MutexWrapper<T> {
    pub fn new(t: T) -> MutexWrapper<T>{
        MutexWrapper {
            0: Arc::new(Mutex::new(t))
        }
    }
}

#[derive(Debug,Serialize,Deserialize,Clone)]
pub struct Node {
    depth: u32,

    embeddings: Vec<f64>,
    url: String,

    #[serde(default, with = "mutexwrapper_serde")]
    node_a: Option<MutexWrapper<Node>>,
    node_a_dist: f64,
    #[serde(default, with = "mutexwrapper_serde")]
    node_b: Option<MutexWrapper<Node>>,
    node_b_dist: f64,
}

impl Node {

    pub fn new(depth: u32, embeddings: Vec<f64>, url: String) -> MutexWrapper<Node> {
        MutexWrapper::new(Node {
            depth,
            embeddings,
            url,
            node_a: None,
            node_a_dist: 0.0,
            node_b: None,
            node_b_dist: 0.0,
        })
    }

    pub fn to_compact_string(&self) -> String{
        let mut space = String::new();
        for i in 0..self.depth {
            space.push_str("  ");
        }
        let mut node_a = String::new();
        if let Some(node) = &self.node_a {
            if let Ok(node) = node.0.lock() {
                node_a = node.to_compact_string();
            }
        }
        let mut node_b = String::new();
        if let Some(node) = &self.node_b {
            if let Ok(node) = node.0.lock() {
                node_b = node.to_compact_string();
            }
        }

        format!("{}\n{}node_a:\n{}{}\n{}node_b:\n{}{}", self.url, space,
                space, node_a, space, space, node_b) 
    }

    pub fn get_url(&self) -> String {
        format!("{}", self.url)
    }

    pub fn traverse(&self, tally: &mut u32, embeddings: &Vec<f64>, mut search_results: Vec<(f64, String, u32)>, threashold: (usize, f64)) -> Vec<(f64, String, u32)> {
        *tally += 1;
        let dist = Node::cosine_sim(&self.embeddings, &embeddings);
        if dist > threashold.1 { 
            search_results.push((dist, self.get_url(), tally.clone()));
        }
        if search_results.len() < threashold.0 {
            if let Some(node_a) = &self.node_a {
                if let Ok(node_a) = node_a.0.lock() {
                    search_results = node_a.traverse(tally, embeddings, search_results, threashold);
                }
            }
            if let Some(node_b) = &self.node_b {
                if let Ok(node_b) = node_b.0.lock() {
                    search_results = node_b.traverse(tally, embeddings, search_results, threashold);
                }
            }
        }

        search_results.sort_by(|a,b|{a.0.total_cmp(&b.0)});

        search_results
    }

    pub fn add_child(&mut self, embeddings: Vec<f64>, url: String) {
        if self.embeddings.len()==0 {
            self.embeddings = embeddings;
            self.url = url;
        } else {
            if self.node_a.is_none() {
                self.node_a_dist = Node::cosine_sim(&self.embeddings, &embeddings);
                self.node_a = Some(Node::new(self.depth+1, embeddings, url));
            } else if self.node_b.is_none() {
                self.node_b_dist = Node::cosine_sim(&self.embeddings, &embeddings);
                self.node_b = Some(Node::new(self.depth+1, embeddings, url));
            } else {
                let mut a_dist = 0.0;
                let mut b_dist = 0.0;
                if let Some(node_a) = &self.node_a {
                    if let Ok(node_a) = node_a.0.lock() {
                        a_dist = Node::cosine_sim(&node_a.embeddings, &embeddings);
                    }
                }
                if let Some(node_b) = &self.node_b {
                    if let Ok(node_b) = node_b.0.lock() {
                        b_dist = Node::cosine_sim(&node_b.embeddings, &embeddings);
                    }
                }
                if a_dist < b_dist {
                    if let Some(node_a) = &self.node_a {
                        if let Ok(mut node_a) = node_a.0.lock() {
                            node_a.add_child(embeddings, url);
                        };
                    }
                } else {
                    if let Some(node_b) = &self.node_b {
                        if let Ok(mut node_b) = node_b.0.lock() {
                            node_b.add_child(embeddings, url);
                        };
                    }
                }
            }
        }
    }
    pub fn cosine_sim(points_a: &Vec<f64>, points_b: &Vec<f64>) -> f64 {
        let dot_prod: f64 = points_a
            .par_iter()
            .zip(points_b.par_iter())
            .map(|(i, j)| i*j)
            .sum();

        let mag_a = f64::sqrt(points_a
            .par_iter()
            .map(|i| i.powi(2))
            .sum());
        let mag_b = f64::sqrt(points_b
            .par_iter()
            .map(|i| i.powi(2))
            .sum());
        
        let norm = mag_a * mag_b;

        dot_prod/norm
    }

    pub fn search(&self, 
                   threashold: f64, 
                   search_size: usize, 
                   search_term: &Vec<f64>,
                   ) -> Vec<(f64, String, u32)> 
    {
        let mut search_results: Vec<(f64, String, u32)> = Vec::new();
        let mut tally = 0;
        search_results = self.traverse(&mut tally, search_term, search_results, (search_size,threashold));

        search_results
    }

    pub fn save_to_file(&self, file_name: String) -> Option<NodeError>{
        match serde_json::to_string(&self.clone()) {
            Ok(content) => {
                match std::fs::write(file_name, content) {
                    Ok(_) => {None},
                    Err(err_msg) => {Some(NodeError{msg: err_msg.to_string()})}
                }
            },
            Err(err_msg) => {Some(NodeError{msg: err_msg.to_string()})}
        }
    }

    pub fn load_model(file_name: String) -> Result<Node, NodeError> {
        match std::fs::read_to_string("./search_model.json") {
            Ok(parent_node) => {
                match serde_json::from_str(&parent_node) {
                    Ok(parent_node) => { Ok(parent_node) },
                    Err(err_msg) => { Err(NodeError{ msg: err_msg.to_string() }) }
                }
            },
            Err(err_msg) => {Err(NodeError{msg: err_msg.to_string()})}
        }
    }
}



pub fn get_openai_embeddings(search_term: String) -> Result< Vec<f64>, NodeError> {
    let chat_request = gpt35!(
        system!("rewrite the following into a good search query to search the api reference documents: "),
        user!(search_term)
        ).get();

    match chat_request {
        Ok(chat_request) => {
            let choice = chat_request.default_choice();
            println!("{}", choice);
            let embeddings = EmbeddingRequest::new(choice).get();
            match embeddings {
                Ok(embeddings) => {
                    match embeddings.get_embeddings() {
                        Some(embeddings) => {Ok(embeddings.clone())},
                        None => {Err(NodeError::from("No search embeddings were found")) }
                    }
                },
                Err(err_msg) => { Err(NodeError { msg: err_msg.message })}
            }
        },
        Err(err_msg) => {Err(NodeError{ msg: err_msg.message})}
    }
}

