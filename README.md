# ColBERT.jl: Efficient, late-interaction retrieval systems in Julia!

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://codetalker7.github.io/ColBERT.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://codetalker7.github.io/ColBERT.jl/dev/)
[![Build Status](https://github.com/codetalker7/ColBERT.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/codetalker7/ColBERT.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/codetalker7/ColBERT.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/codetalker7/ColBERT.jl)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor%27s%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

[ColBERT.jl](https://codetalker7/colbert.jl) is a pure Julia package for the ColBERT information retrieval system[^1][^2][^3], allowing developers
to integrate this powerful neural retrieval algorithm into their own downstream tasks. ColBERT (**c**ontextualized **l**ate interaction over **BERT**) has emerged as a state-of-the-art approach for efficient and effective document retrieval, thanks to its ability to leverage contextualized embeddings from pre-trained language models like BERT.

[Inspired from the original Python implementation of ColBERT](https://github.com/stanford-futuredata/ColBERT), with [ColBERT.jl](https://codetalker7/colbert.jl), you can now bring this capability to your Julia applications, whether you're working on natural language processing tasks, information retrieval systems, or other areas where relevant document retrieval is crucial. Our package provides a simple and intuitive interface for using ColBERT in Julia, making it easy to get started with this powerful algorithm.

## Get Started

### Dataset and preprocessing

This package is currently under active development, and has not been registered in Julia's package registry yet. To develop this package, simply clone this repository, and from the root of the package just `dev` it:

```julia
julia> ] dev .
```

We'll go through an example of the `lifestyle/dev` split of the [LoTTe dataset](https://github.com/stanford-futuredata/colbert/blob/main/lotte.md). To download the dataset, you can use the `examples/lotte.sh` script. We'll work with the first `1000` documents of the dataset:

```
$ cd examples
$ ./lotte.sh
$ head -n 1000 downloads/lotte/lifestyle/dev/collection.tsv > 1kcollection.tsv
$ wc -l 1kcollection.tsv
1000 1kcollection.txt
```

The `1kcollection.tsv` file has documents in the format `pid \t <document text>`, where `pid` is the unique ID of the document. For now, the package only supports collections which have one document per line. So, we'll simply remove the `pid` from each document in `1kcollection.tsv`, and save the resultant file of documents in `1kcollection.txt`. Here's a simple Julia script you can use to do this preprocessing using the [`CSV.jl`](https://github.com/JuliaData/CSV.jl) package:

```julia
using CSV
file = CSV.File("1kcollection.tsv"; delim = '\t', header = [:pid, :text],
        types = Dict(:pid => Int, :text => String), debug = true, quoted = false)
for doc in file.text
    open("1kcollection.txt", "a") do io
        write(io, doc*"\n")
    end
end
```

We now have our collection of documents to index!

### The `ColBERTConfig`

The next step is to create a configuration object containing details about all parameters used during indexing/searching using ColBERT. All this information is contained in a type called `ColBERTConfig`. Creating a `ColBERTConfig` is easy; it has the right defaults for most users, and one can change the settings using simple kwargs. In this example, we'll create a config for the collection `1kcollection.txt` we just created, and we'll also use [`CUDA.jl`](https://github.com/JuliaGPU/CUDA.jl) for GPU support (you can use any GPU backend supported by [Flux.jl](https://github.com/FluxML/Flux.jl))!

```julia
julia>  using ColBERT, CUDA, Random;

julia>  Random.seed!(0)                     # global seed for a reproducible index

julia>  config = ColBERTConfig(
            use_gpu = true,
            collection = "./1kcollection.txt",
            doc_maxlen = 300,               # max length beyond which docs are truncated
            index_path = "./1kcollection_index/",
            chunksize = 200                 # number of docs to store in a chunk
        );
```

You can read more about a [`ColBERTConfig`](https://github.com/codetalker7/ColBERT.jl/blob/302b68caf0c770b5e23c83b1f204808185ffaac5/src/infra/config.jl#L1) from it's docstring.

### Building the index

Building the index is even easier than creating a config; just build an `Indexer` and call the `index` function. I used an NVIDIA GeForce RTX 2020 Ti card to build the index:

```julia
julia>  indexer = Indexer(config);

julia>  @time index(indexer)
[ Info: Saving metadata to ./1kcollection_index/3.metadata.json
[ Info: Encoding 200 passages.
[ Info: Saving chunk 4:          200 passages and 49415 embeddings. From passage #601 onward.
[ Info: Saving compressed codes to ./1kcollection_index/4.codes.jld2 and residuals to ./1kcollection_index/4.residuals.jld2
[ Info: Saving doclens to ./1kcollection_index/doclens.4.jld2
[ Info: Saving metadata to ./1kcollection_index/4.metadata.json
[ Info: Encoding 200 passages.
[ Info: Saving chunk 5:          200 passages and 52304 embeddings. From passage #801 onward.
[ Info: Saving compressed codes to ./1kcollection_index/5.codes.jld2 and residuals to ./1kcollection_index/5.residuals.jld2
[ Info: Saving doclens to ./1kcollection_index/doclens.5.jld2
[ Info: Saving metadata to ./1kcollection_index/5.metadata.json
[ Info: Running some final checks.
[ Info: Checking if all files are saved.
[ Info: Found all files!
[ Info: Collecting embedding ID offsets.
[ Info: Saving the indexing metadata.
[ Info: Building the centroid to embedding IVF.
[ Info: Loading codes for each embedding.
[ Info: Sorting the codes.
[ Info: Getting unique codes and their counts.
[ Info: Saving the IVF.
397.647997 seconds (137.12 M allocations: 34.834 GiB, 7.85% gc time, 11.96% compilation time: <1% of which was recompilation)
```

### Searching

Once you've built the index for your collection of docs, it's now time to perform a query search. This involves creating a `Searcher` from the path of the index:

```julia
julia>  using ColBERT, CUDA;

julia>  searcher = Searcher("1kcollection_index");
```

Next, simply feed a query to the `search` function, and get the top-`k` best documents for your query:

```julia
julia>  query = "what is 1080 fox bait poisoning?";

julia>  @time pids, scores = search(searcher, query, 10)
0.425458 seconds (3.51 M allocations: 430.293 MiB, 13.95% gc time, 1.12% compilation time)
([999, 383, 378, 386, 547, 384, 385, 963, 323, 344], Float32[8.543619, 7.804471, 7.039251, 6.7534733, 6.523997, 6.1977453, 6.131935, 6.086709, 6.0386653, 5.7597084])
```

You can now use these `pids` to see which documents match the best against your query:

```julia
julia> print(readlines("1kcollection.txt")[pids[1]])
Tl;dr - Yes, it sounds like a possible 1080 fox bait poisoning. Can't be sure though. The traditional fox bait is called 1080. That poisonous bait is still used in a few countries to kill foxes, rabbits, possums and other mammal pests. The toxin in 1080 is Sodium fluoroacetate. Wikipedia is a bit vague on symptoms in animals, but for humans they say: In humans, the symptoms of poisoning normally appear between 30 minutes and three hours after exposure. Initial symptoms typically include nausea, vomiting and abdominal pain; sweating, confusion and agitation follow. In significant poisoning, cardiac abnormalities including tachycardia or bradycardia, hypotension and ECG changes develop. Neurological effects include muscle twitching and seizures... One might safely assume a dog, especially a small Whippet, would show symptoms of poisoning faster than the 30 mins stated for humans. The listed (human) symptoms look like a good fit to what your neighbour reported about your dog. Strychnine is another commonly used poison against mammal pests. It affects the animal's muscles so that contracted muscles can no longer relax. That means the muscles responsible of breathing cease to operate and the animal suffocates to death in less than two hours. This sounds like unlikely case with your dog. One possibility is unintentional pet poisoning by snail/slug baits. These baits are meant to control a population of snails and slugs in a garden. Because the pelletized bait looks a lot like dry food made for dogs it is easily one of the most common causes of unintentional poisoning of dogs. The toxin in these baits is Metaldehyde and a dog may die inside four hours of ingesting these baits, which sounds like too slow to explain what happened to your dog, even though the symptoms of this toxin are somewhat similar to your case. Then again, the malicious use of poisons against neighbourhood dogs can vary a lot. In fact they don't end with just pesticides but also other harmful matter, like medicine made for humans and even razorblades stuck inside a meatball, have been found in baits. It is quite impossible to say what might have caused the death of your dog, at least without autopsy and toxicology tests. The 1080 is just one of the possible explanations. It is best to always use a leash when walking dogs in populated areas and only let dogs free (when allowed by local legislation) in unpopulated parks and forests and suchlike places.
```

## Key Features

As of now, the package supports the following:

  - Offline indexing of documents using embeddings generated from the `"colbert-ir/colbertv2.0"` (or any other checkpoint supported by [`Transformers.jl`](https://github.com/chengchingwen/Transformers.jl)) HuggingFace checkpoint.
  - Compression/decompression based on the ColBERTv2[^2] compression scheme, i.e using $k$-means centroids and quantized residuals.
  - A simple searching/ranking module, which is used to get the top `k`-ranked documents for a query by computing MaxSim[^1] scores.
  - GPU support using any backend supported by [Flux.jl](https://github.com/FluxML/Flux.jl), both for indexing and searcher.

## Contributing

Though the package is in it's early stage, PRs and issues are always welcome! Stay tuned for docs and relevant contribution related information to be added to the repo.

## Stay Tuned

We're excited to continue developing and improving [ColBERT.jl](https://github.com/codetalker7/ColBERT.jl), with the following components to be added soon (in no order of priority):

  - A training module, to be used to pre-train a ColBERT model from scratch.
  - Adding support for multiple GPUs. Currently the package is designed to support only on GPU.
  - Implementation of multiprocessing and distributed training.
  - More utilities to the indexer, like updating/removing documents from the index.
  - PLAID[^3] optimizations.
  - More documentation! The package needs a lot more documentation and examples.
  - Integration with downstream packages like [AIHelpMe.jl](https://github.com/svilupp/AIHelpMe.jl) and [PromptingTools.jl](https://github.com/svilupp/PromptingTools.jl). This package can be used as a backend for any information retrieval task.
  - Add support for optimization tricks like [vector pooling](https://www.answer.ai/posts/colbert-pooling.html).

## Cite us!

If you find this package to be useful in your research/applications, please cite the package:

    @misc{ColBERT.jl,
        author  = {Siddhant Chaudhary <urssidd@gmail.com> and contributors},
        title   = {ColBERT.jl},
        url     = {https://github.com/codetalker7/ColBERT.jl},
        version = {v0.1.0},
        year    = {2024},
        month   = {5}
    }

[^1]: [ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT](https://arxiv.org/abs/2004.12832) (SIGIR'20)
[^2]: [ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction](https://arxiv.org/abs/2112.01488) (NAACL'22).
[^3]: [PLAID: An Efficient Engine for Late Interaction Retrieval](https://arxiv.org/abs/2205.09707) (CIKM'22).
