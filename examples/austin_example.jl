using ColBERT
using Test
using Random
using CSV

## Convert to TSV
# fn = "cityofaustin.csv"
# df = CSV.File(joinpath(@__DIR__, fn))
# CSV.write(joinpath(@__DIR__, "cityofaustin.tsv"), df; delim = '\t')

## Load manually as CSV
fn = joinpath(@__DIR__, "cityofaustin.csv")
file = CSV.File(fn; header = [:pid, :text], types = Dict(:pid => Int, :text => String))
collection = Collection(fn, file.text[1:10])
length(collection.data)

nbits = 2   # encode each dimension with 2 bits
doc_maxlen = 300   # truncate passages at 300 tokens

# checkpoint = "colbert-ir/"                       # the HF checkpoint
checkpoint = "colbert-ir/colbertv2.0"                       # the HF checkpoint
index_root = "indexes"
index_name = "austin_$(nbits)bits"
index_path = joinpath(index_root, index_name)

config = ColBERTConfig(
    RunSettings(
        experiment = "notebook",
    ),
    TokenizerSettings(),
    ResourceSettings(
        checkpoint = checkpoint,
        collection = collection,
        index_name = index_name
    ),
    DocSettings(
        doc_maxlen = doc_maxlen,
    ),
    QuerySettings(),
    IndexingSettings(
        index_path = index_path,
        index_bsize = 3,
        nbits = nbits,
        kmeans_niters = 20
    ),
    SearchSettings()
)

# create and run the indexer
indexer = Indexer(config)
@time index(indexer)
ColBERT.save(config)

# # Searching

# create the config
checkpoint = "colbert-ir/colbertv2.0"                       # the HF checkpoint
index_root = "indexes"
index_name = "austin_$(nbits)bits"
index_path = joinpath(index_root, index_name)

# build the searcher
searcher = Searcher(index_path)

# search for a query
query = "what are white spots on raspberries?"
pids, scores = search(searcher, query, 2)
print(searcher.config.resource_settings.collection.data[pids])

query = "are rabbits easy to housebreak?"
pids, scores = search(searcher, query, 9)
print(searcher.config.resource_settings.collection.data[pids])
