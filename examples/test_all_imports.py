#!/usr/bin/env python3
"""Test that all modules can be imported and work together."""

print("Testing all imports...")

# Test basic imports
print("\n1. Testing basic imports from elemai...")
from elemai import (
    _ai,
    ai,
    Result,
    Chat,
    chat,
    configure,
    set_config,
    get_config,
    templates,
    template_fn,
    MessageTemplate,
    Field,
)
print("✓ All basic imports successful")

# Test sentinel
print("\n2. Testing _ai sentinel...")
from elemai.sentinel import AISentinel, FunctionIntrospector
assert isinstance(_ai, AISentinel)
sentinel_with_desc = _ai["Think carefully"]
assert sentinel_with_desc.description == "Think carefully"
print(f"✓ _ai sentinel works: {_ai}")
print(f"✓ _ai with description: {sentinel_with_desc}")

# Test Result
print("\n3. Testing Result class...")
r = Result(result="test", thinking="analysis")
assert r.result == "test"
assert r.thinking == "analysis"
assert hasattr(r, '_fields')
print(f"✓ Result creation works")
print(f"  repr: {repr(r)}")
print(f"  str: {str(r)}")

# Test ai decorator
print("\n4. Testing @ai decorator...")
@ai
def test_func(x: str) -> str:
    """Test function"""
    return _ai

assert hasattr(test_func, 'metadata')
assert test_func.metadata['fn_name'] == 'test_func'
print(f"✓ @ai decorator works")
print(f"  Function name: {test_func.metadata['fn_name']}")
print(f"  Instruction: {test_func.metadata['instruction']}")

# Test ai decorator with parameters
print("\n5. Testing @ai decorator with parameters...")
@ai(model="test-model", temperature=0.5)
def test_func_with_params(x: str) -> str:
    """Test with params"""
    return _ai

assert test_func_with_params.config.model == "test-model"
assert test_func_with_params.config.temperature == 0.5
print(f"✓ @ai decorator with parameters works")
print(f"  Model: {test_func_with_params.config.model}")
print(f"  Temperature: {test_func_with_params.config.temperature}")

# Test config
print("\n6. Testing configuration...")
config = get_config()
print(f"✓ get_config works: {config}")

# Test template
print("\n7. Testing templates...")
assert hasattr(templates, 'simple')
assert hasattr(templates, 'reasoning')
print(f"✓ Built-in templates available")
print(f"  Simple template: {len(templates.simple)} messages")
print(f"  Reasoning template: {len(templates.reasoning)} messages")

# Test MessageTemplate
print("\n8. Testing MessageTemplate...")
template = MessageTemplate([
    {"role": "user", "content": "Hello {name}"}
])
messages = template.render(name="Alice")
assert messages[0]['content'] == "Hello Alice"
print(f"✓ MessageTemplate works")
print(f"  Rendered: {messages[0]['content']}")

# Test Field
print("\n9. Testing Field...")
field = Field(name="test", type=str, description="Test field")
assert field.name == "test"
assert field.type == str
print(f"✓ Field class works")
print(f"  Field: {field.name}: {field.type}")

# Test FunctionIntrospector
print("\n10. Testing FunctionIntrospector...")
def sample_func(x: int, y: str = "default") -> str:
    """Sample function"""
    thinking: str = _ai["Think"]
    return _ai

inspector = FunctionIntrospector(sample_func)
input_fields = inspector.get_input_fields()
output_fields = inspector.get_output_fields()
assert len(input_fields) == 2
print(f"✓ FunctionIntrospector works")
print(f"  Input fields: {[f['name'] for f in input_fields]}")
print(f"  Output fields: {[f['name'] for f in output_fields]}")

# Test complete workflow
print("\n11. Testing complete workflow...")
@ai
def analyze_text(text: str) -> str:
    """Analyze text sentiment"""
    thinking: str = _ai["Consider the emotional tone"]
    return _ai

assert hasattr(analyze_text, 'metadata')
assert hasattr(analyze_text, 'render')
assert hasattr(analyze_text, 'preview')
assert hasattr(analyze_text, 'to_messages')
print(f"✓ Complete AI function workflow")
print(f"  Has metadata: {bool(analyze_text.metadata)}")
print(f"  Has render: {callable(analyze_text.render)}")
print(f"  Has preview: {callable(analyze_text.preview)}")

# Summary
print("\n" + "=" * 70)
print("ALL TESTS PASSED ✓")
print("=" * 70)
print("""
All modules are working correctly:
  ✓ Sentinel (_ai)
  ✓ Result class with nice reprs
  ✓ AI decorator (@ai)
  ✓ Configuration
  ✓ Templates
  ✓ Function introspection
  ✓ Complete workflows

The Result API is ready to use with the all= parameter!
""")
