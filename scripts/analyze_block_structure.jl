using JSON3

# Function to count block openings and closings
function analyze_block_structure(file_path)
    content = read(file_path, String)
    lines = split(content, '\n')
    
    block_starts = ["if", "for", "while", "function", "module", "struct", "begin", "let", "try", "do"]
    block_ends = ["end"]
    
    stack = []
    block_count = 0
    
    for (line_num, line) in enumerate(lines)
        # Remove comments
        line = split(line, '#')[1]
        
        # Check for block starts
        for block in block_starts
            if occursin(block * " ", line) || endswith(line, block)
                block_count += 1
                push!(stack, (block, line_num))
                println("Block start: $block at line $line_num")
            end
        end
        
        # Check for block ends
        if occursin("end", line)
            block_count -= 1
            if !isempty(stack)
                start_block, start_line = pop!(stack)
                println("Block end at line $line_num (matches $(start_block) from line $start_line)")
            else
                println("WARNING: Unmatched end at line $line_num")
            end
        end
    end
    
    println("\nFinal block count: $block_count")
    if !isempty(stack)
        println("\nUnclosed blocks:")
        for (block, line) in stack
            println("- $block started at line $line")
        end
    end
end

# Analyze the tokenizer file
analyze_block_structure("/home/ubuntu/ModernBert.jl/src/minimal_tokenizer.jl")
