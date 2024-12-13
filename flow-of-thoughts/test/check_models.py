from language_models.chatgpt import ChatGPT
from language_models.llamachat_hf import LlamaHF
from language_models.abstract_language_model import AbstractLanguageModel

def check_models(model: AbstractLanguageModel):
    print(model.query("What is the meaning of life?"))

from thoughts.operations import Generate, Parser
from thoughts.thought import Thought

# Test Generate operation with multiple responses
def test_generate_operation(model: AbstractLanguageModel):
    # Create a test parent thought
    parent = Thought("Test parent thought", [], [], is_executable=True)
    
    # Create a simple test prompt
    test_prompt = Parser(prompt="Generate a response", name="test_prompt")
    
    # Test with different numbers of responses
    for num_responses in [1, 2, 3]:
        print(f"\nTesting Generate operation with {num_responses} responses:")
        
        # Create Generate operation
        gen_op = Generate(
            parents_thoughts=[parent],
            language_model=model,
            branching_factor=num_responses,
            generate_prompt=test_prompt
        )
        
        # Execute generation
        gen_op.generate_children()
        
        # Verify number of children thoughts
        children = gen_op.get_children_thoughts()
        print(f"Requested {num_responses} responses, got {len(children)} children thoughts")
        
        # Print generated responses
        for i, child in enumerate(children):
            print(f"Response {i+1}: {child.content}")
            print(f"Is executable: {child.is_executable}")

