using AIHelpMe
using PromptingTools
const PT = PromptingTools
using PromptingTools.Experimental.RAGTools
const RT = PromptingTools.Experimental.RAGTools
using JSON3, Dates, Statistics

# Load all chunks quickly -- defaults to 1024 Bool
AIHelpMe.update_pipeline!(:gold)
index = AIHelpMe.load_index!([
    :julia, :juliadata, :tidier, :sciml, :plots, :makie, :genie])

# index.chunks

# Load the evaluation set
fn = "benchmark/dataframe_combined_filtered-qa-evals.json"
eval_set = JSON3.read(fn)

cfg = AIHelpMe.RAG_CONFIG.retriever;
# Disable reranker for dumb OAI comparison
cfg.reranker = RT.NoReranker()
kwargs = AIHelpMe.RAG_KWARGS.retriever_kwargs;

# Test retrieval
res = RT.retrieve(cfg, index, "How do I install packages?"; kwargs...)
res.context

# Function to process a single evaluation item
function process_eval_item(eval_item)
    global cfg, kwargs

    query = eval_item.question
    correct_chunk = eval_item.context

    try
        # Perform retrieval
        t = @elapsed result = RAGTools.retrieve(cfg, index, query; kwargs...)

        # Check if the correct chunk is in the result
        dist = PT.distance_longest_common_subsequence(
            correct_chunk, result.context)
        hit = any(dist .<= 0.33)
        rank = findfirst(dist .<= 0.33)

        return (;
            query = query,
            correct_chunk = correct_chunk,
            retrieved_chunks = result.context,
            hit,
            rank,
            elapsed = t,
            error = nothing
        )
    catch e
        @warn "Retrieval failed for query: $query; $e"
        return (
            query = query,
            correct_chunk = correct_chunk,
            retrieved_chunks = String[],
            hit = missing,
            rank = missing,
            elapsed = missing,
            error = sprint(showerror, e)
        )
    end
end

function report_results(results)
    # Calculate and print summary statistics
    total_queries = length(results)
    successful_queries = count(r -> isnothing(r.error), results)
    success_results = filter(r -> isnothing(r.error), results)
    hits = count(r -> r.hit, success_results)
    accuracy = hits / length(success_results)
    mean_reciprocal_rank = mean(isnothing(r.rank) ? 0.0 : 1 / r.rank
    for r in success_results)
    avg_time = mean(r.elapsed for r in results)
    success_rate = successful_queries / total_queries

    println("\nBenchmark Summary:")
    println(" - Total queries: ", total_queries)
    println(" - Successful queries: ", successful_queries)
    println(" - Failed queries: ", total_queries - successful_queries)
    println(" - Success rate: ", round(success_rate * 100, digits = 2), "%")
    println(" - Hits: ", hits)
    println(" - Accuracy: ", round(accuracy * 100, digits = 2), "%")
    println(" - Average time: ", round(avg_time, digits = 2), "s")
    println(
        " - Mean reciprocal rank: ", round(mean_reciprocal_rank, digits = 2))
end

# Run the benchmark asynchronously
@time results = asyncmap(process_eval_item, eval_set; ntasks = 30);
report_results(results)

# # All eval files

fn = ["benchmark/dataframe_combined_filtered-qa-evals.json",
    "benchmark/makie_combined_filtered-qa-evals.json",
    "benchmark/tidier_combined_filtered-qa-evals.json",
    "benchmark/sciml_combined_filtered-qa-evals.json",
    "benchmark/plots_combined_filtered-qa-evals.json"
]

evals = vcat([JSON3.read(f) for f in fn]...)

# Run the benchmark asynchronously
@time results = asyncmap(process_eval_item, evals; ntasks = 30);
report_results(results)

# Hits @ top5
# Benchmark Summary:
#  - Total queries: 165
#  - Successful queries: 165
#  - Failed queries: 0
#  - Success rate: 100.0%
#  - Hits: 145
#  - Accuracy: 87.88%
#  - Average time: 0.71s
#  - Mean reciprocal rank: 0.76

# Update to top 10 items
kwargs = (; kwargs..., top_n = 10);
@time results = asyncmap(process_eval_item, evals; ntasks = 30);
report_results(results)

# Hits @ top10
# Benchmark Summary:
#  - Total queries: 165
#  - Successful queries: 165
#  - Failed queries: 0
#  - Success rate: 100.0%
#  - Hits: 151
#  - Accuracy: 91.52%
#  - Average time: 1.23s
#  - Mean reciprocal rank: 0.76

# # Performance with re-ranker

# Reranker + top-5
cfg.reranker = RT.CohereReranker();
kwargs = AIHelpMe.RAG_KWARGS.retriever_kwargs;

# Run the benchmark asynchronously
@time results = asyncmap(process_eval_item, evals; ntasks = 30);
report_results(results)

# Hits @ top5 with reranker
# Benchmark Summary:
#  - Total queries: 165
#  - Successful queries: 165
#  - Failed queries: 0
#  - Success rate: 100.0%
#  - Hits: 151
#  - Accuracy: 91.52%
#  - Average time: 1.03s
#  - Mean reciprocal rank: 0.86