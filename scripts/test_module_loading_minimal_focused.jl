using Pkg
Pkg.activate(".")

# Test minimal module loading with timing
@time begin
    # Try loading just the module
    using ModernBert
    println("Base module loaded successfully")
end
