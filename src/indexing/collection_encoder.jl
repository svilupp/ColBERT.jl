using ..ColBERT: ColBERTConfig

struct CollectionEncoder
    config::ColBERTConfig
    checkpoint::Checkpoint
end

function encode_passages(encoder::CollectionEncoder, passages::Vector{String})
    @info "Encoding $(length(passages)) passages."

    # TODO: complete this implementation!
end
