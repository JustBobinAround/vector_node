# Vector Node

The Vector Node is a Rust implementation of a tree data structure designed
to organize web pages based on their content embeddings. It provides a
hierarchical structure where each node represents a web page, and the tree is
built based on the cosine similarity between content embeddings. The tree is 
sorted by closer embeddings, which should help find relevant results faster.
At some point, this will be changed to use ANN for search instead.

## Features

- **Tree Structure**: Each node in the tree contains information about a web
  page, including depth, embeddings, URL, and page number.
- **Binary Tree**: The tree follows a binary structure, with each node having
  two children (`node_a` and `node_b`). This should improve finding results
  earlier, but the time complexity is still O(n).
- **Multithreading**: The code uses the `rayon` crate for parallel processing
  to improve performance of the cosine similarity calculations.
- **Serialization and Deserialization**: The tree can be serialized to and
  deserialized from JSON format.

## Usage

### Creating a Node and Adding a Child

```rust
use vector_node::prelude::*;

let parent_node = Node::new(0, embeddings, url, page);

println!("Embedding Progress: {}/{}", count, db_len);
if let Ok(mut parent_node) = parent_node.0.lock() {
    parent_node.add_child(embeddings, url, page);
}

```

### Searching the Tree

```rust
let search_term = "search query".to_string();
let min_similarity = 0.8; 
let max_search_results = 5;
let search_results = root_node.search(min_similarity, max_search_results, search_term, get_openai_embeddings);
// you may add your own embeddings closure if you want to add your own type of embeddings
```

### Saving and Loading the Node Structure

```rust
root_node.save_to_file("search_tree.json".to_string());

let loaded_tree = Node::load_model("search_tree.json".to_string()).unwrap();
```

## Dependencies

- [rayon](https://crates.io/crates/rayon): For parallel processing.
- [serde](https://crates.io/crates/serde): For serialization and deserialization.
- [openai_api](https://crates.io/crates/openai_api): For interacting with the OpenAI API.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
