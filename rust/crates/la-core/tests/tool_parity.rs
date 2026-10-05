//! Tool-registry parity snapshot: the Rust registry order must match
//! `_default_tools` in `src/live_ai.py` (checked-in snapshot generated from
//! the reference Python code; update deliberately).

use la_core::tools::defaults::registry_for_tests;

/// The reference tool order (Python `_default_tools`, minus the two worker
/// lifecycle tools' python-only name variants — see the notes below).
const SNAPSHOT: &[&str] = &[
    "model_info",
    "recall_personal",
    "store_personal",
    "clear_chat",
    "image_response",
    "allow_speech",
    "summarize_doc",
    "query_document",
    "image_search",
    "list_corpora",
    "store_corpus",
    "query_corpus",
    "capture_image",
    "system_info",
    "current_date_time",
    "get_working_directory",
    "run_command",
    "search_in_file",
    "search_pattern_in_files",
    "read_text_file",
    "replace_string_in_file",
    "write_text_file",
    "web_search",
    "webpage_content",
    "webpage_headlines",
    "download_file",
    "extract_archive",
    "zip_files",
    "fetch_financial_data",
    "save_financial_chart",
    "generate_image",
    "unload_image_generator",
    "exit_gracefully",
];

#[test]
fn tool_names_and_order_match_python() {
    let registry = registry_for_tests();
    let names = registry.names();
    assert_eq!(
        names.len(),
        SNAPSHOT.len(),
        "tool count diverged: got {names:?}"
    );
    for (index, expected) in SNAPSHOT.iter().enumerate() {
        assert_eq!(
            names[index], *expected,
            "tool order diverged at {index} (snapshot order must mirror _default_tools)"
        );
    }
}

#[test]
fn tools_have_descriptions_and_schemas() {
    let registry = registry_for_tests();
    for def in &registry.tools {
        assert!(
            !def.description.is_empty(),
            "{}: missing description",
            def.name
        );
        assert!(
            def.summary().len() > 5,
            "{}: missing summary line",
            def.name
        );
        assert_eq!(
            def.schema.get("type").and_then(|v| v.as_str()),
            Some("object"),
            "{}: schema must be an object",
            def.name
        );
        assert!(
            def.schema.get("properties").is_some(),
            "{}: schema must declare properties",
            def.name
        );
    }
}

#[test]
fn required_params_match_required_lists() {
    let registry = registry_for_tests();
    // spot-check Python-required parameters
    for (name, expected) in [
        ("run_command", vec!["cmd"]),
        ("write_text_file", vec!["file", "content"]),
        ("generate_image", vec!["prompt"]),
        ("query_corpus", vec!["query", "corpus_name"]),
    ] {
        let def = registry.get(name).expect(name);
        let required: Vec<String> = def.required_params().into_iter().map(|(n, _)| n).collect();
        assert_eq!(required, expected, "{name}: required params");
    }
    // optional params: schema properties with a default
    let def = registry.get("write_text_file").unwrap();
    assert!(
        def.schema["properties"]["overwrite"]
            .get("default")
            .is_some()
    );
}
