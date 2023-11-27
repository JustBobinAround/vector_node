# Vector Node

The Vector Node is a Rust implementation of a K-D tree structure designed
to organize web pages based on their vector embeddings. It provides a
hierarchical structure where each node represents a web page, and the tree is
built based on the cosine similarity between content embeddings. The tree is 
sorted by closer embeddings, which should help find relevant results faster.
At some point, this will be changed to use ANN for search instead.

## Features

- **K-D Tree Structure**: Each node in the K-D tree contains information about a,
  including depth, embeddings, URL, and page number. The tree follows a binary structure,
  with each node having two children (`node_a` and `node_b`). This should improve finding results
  to an average time of Log(n), but a worst case time complexity of O(n).
- **Multithreading**: This repo uses the `rayon` crate for parallel processing
  to improve performance of the cosine similarity calculations.
- **Serialization and Deserialization**: The tree can be serialized to and
  deserialized from JSON format.
- **Complementary Prompting Function:** This library includes an example 
  function on how to build a search closure.

## Usage

### Creating a Node and Adding a Child

```rust
use vector_node::prelude::*;

let parent_node = Node::new(0, embeddings, url, page);

if let Ok(mut parent_node) = parent_node.0.lock() {
    parent_node.add_child(embeddings, url, page);
}

```

### Searching the Tree

```rust
let search_term = get_openai_embeddings("search query".to_string());
let min_similarity = 0.8; 
let max_search_results = 5;
let search_results = root_node.search(min_similarity, max_search_results, search_term);
// you may add your own search closure if you want to add your own type of embeddings
```

### Example Embedding

```rust
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
```

### Saving and Loading the Node Structure

```rust
root_node.save_to_file("search_tree.json".to_string());

let loaded_tree = Node::load_model("search_tree.json".to_string()).unwrap();
```

## Dependencies

- [rayon](https://crates.io/crates/rayon): For parallel processing.
- [serde](https://crates.io/crates/serde): For serialization and deserialization.
- [openai_api](https://github.com/JustBobinAround/openai_api): Library I made for interacting with the OpenAI API.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
