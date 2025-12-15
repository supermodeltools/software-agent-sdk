# Claude Code Alignment for Skill Package Loading

## Overview

The OpenHands SDK now supports loading skill packages with both:
1. **manifest.json** (Claude Desktop Extensions-aligned format)
2. **skill-package.yaml** (legacy format)

This enables cross-platform package distribution and better integration with the Model Context Protocol (MCP) ecosystem.

## Changes to package_loader.py

### New Helper Function

```python
def _load_descriptor(package_module) -> dict[str, Any]:
    """Load package descriptor, trying manifest.json first, then skill-package.yaml."""
```

This function:
1. Attempts to load `manifest.json` from the package
2. Falls back to `skill-package.yaml` if JSON not found
3. Raises FileNotFoundError if neither file exists

### Updated Functions

All package loading functions now use `_load_descriptor()`:
- `list_skill_packages()` - Lists all installed skill packages
- `get_skill_package()` - Gets a specific package by name
- `load_skills_from_package()` - Loads skills with format-aware spec extraction

### Format Detection

The `load_skills_from_package()` function auto-detects the descriptor format:

```python
# Handle both formats
if "spec" in descriptor:
    # Old YAML format: skills are in spec.skills
    skills_spec = descriptor.get("spec", {}).get("skills", [])
else:
    # New JSON format: skills are at top level
    skills_spec = descriptor.get("skills", [])
```

## Descriptor Format Comparison

### manifest.json (New Format)

```json
{
  "name": "simple-code-review",
  "version": "1.0.0",
  "displayName": "Simple Code Review",
  "description": "Basic code review skills",
  "skills": [
    {
      "name": "code-review",
      "path": "skills/code_review.md",
      "type": "keyword-triggered",
      "triggers": ["codereview", "review-code"]
    }
  ]
}
```

**Structure**: Flat, with `skills` at the top level

### skill-package.yaml (Legacy Format)

```yaml
apiVersion: openhands.ai/v1
kind: SkillPackage

metadata:
  name: simple-code-review
  version: "1.0.0"
  displayName: "Simple Code Review"
  description: "Basic code review skills"

spec:
  skills:
    - name: code-review
      path: skills/code_review.md
      type: keyword-triggered
      triggers:
        - codereview
        - review-code
```

**Structure**: Nested, with `skills` under `spec.skills`

## Backwards Compatibility

The changes are **fully backwards compatible**:

1. Existing packages with `skill-package.yaml` continue to work
2. New packages can use `manifest.json`
3. Packages can include both files during transition
4. No changes required to existing code that uses package_loader

## Usage Examples

### Loading Packages (No Changes Required)

```python
from openhands.sdk.context.skills.package_loader import (
    list_skill_packages,
    get_skill_package,
    load_skills_from_package
)

# List all packages - works with both formats
packages = list_skill_packages()
for pkg in packages:
    print(f"Package: {pkg['name']}")

# Get specific package - works with both formats  
pkg = get_skill_package('simple-code-review')

# Load skills - works with both formats
repo_skills, knowledge_skills = load_skills_from_package('simple-code-review')
```

### Accessing Descriptor Data

When working directly with descriptors, handle both formats:

```python
pkg = get_skill_package('my-package')
descriptor = pkg['descriptor']

# Handle both nested (YAML) and flat (JSON) structures
if 'metadata' in descriptor:
    # Old YAML format
    metadata = descriptor['metadata']
    skills = descriptor.get('spec', {}).get('skills', [])
else:
    # New JSON format
    metadata = descriptor
    skills = descriptor.get('skills', [])

print(f"Display Name: {metadata.get('displayName', 'Unknown')}")
```

## Creating New Packages

### Recommended Approach: Use manifest.json

1. Create `manifest.json` in your package root:

```json
{
  "name": "my-awesome-skills",
  "version": "1.0.0",
  "displayName": "My Awesome Skills",
  "description": "A collection of useful skills",
  "author": {
    "name": "Your Name",
    "email": "you@example.com"
  },
  "keywords": ["skills", "awesome"],
  "license": "MIT",
  "skills": [
    {
      "name": "my-skill",
      "description": "Description of my skill",
      "path": "skills/my_skill.md",
      "type": "keyword-triggered",
      "triggers": ["myskill", "awesome"]
    }
  ],
  "package": {
    "type": "python",
    "name": "my-awesome-skills",
    "entry_point": "my_awesome_skills"
  }
}
```

2. Include manifest.json in your package data (pyproject.toml):

```toml
[tool.setuptools.package-data]
my_awesome_skills = ["manifest.json", "skills/*.md"]
```

3. Register the entry point:

```toml
[project.entry-points."openhands.skill_packages"]
my-awesome-skills = "my_awesome_skills"
```

## Migration from YAML to JSON

### Option 1: Add JSON alongside YAML (Recommended)

Keep both files for maximum compatibility:

```
my-package/
├── manifest.json          # New format
├── skill-package.yaml     # Old format (for backwards compat)
├── pyproject.toml
└── my_package/
    ├── __init__.py
    ├── manifest.json      # Include in package
    ├── skill-package.yaml # Include in package
    └── skills/
        └── skill.md
```

Update pyproject.toml:
```toml
[tool.setuptools.package-data]
my_package = ["manifest.json", "skill-package.yaml", "skills/*.md"]
```

### Option 2: Replace YAML with JSON

Only do this if you don't need to support old SDK versions:

1. Create `manifest.json` from `skill-package.yaml` content
2. Remove `skill-package.yaml`
3. Update package data to only include manifest.json
4. Test with SDK

## Integration with OpenHands

The OpenHands SDK will automatically:
1. Discover packages via entry points
2. Load the appropriate descriptor format
3. Parse skills from the descriptor
4. Load skill content from markdown files
5. Create Skill objects with triggers

No changes needed to your OpenHands agent code!

## Testing

Test your package works with both formats:

```bash
# Install your package
pip install -e .

# Test discovery
python -c "from openhands.sdk.context.skills.package_loader import list_skill_packages; print(list_skill_packages())"

# Test loading
python -c "from openhands.sdk.context.skills.package_loader import load_skills_from_package; print(load_skills_from_package('your-package-name'))"
```

## Benefits

1. **Cross-Platform**: Packages work with OpenHands SDK and Claude Desktop
2. **Standard Format**: JSON is more widely adopted than custom YAML
3. **MCP Compatible**: Better integration with Model Context Protocol ecosystem
4. **Developer Friendly**: Familiar format similar to npm package.json
5. **No Breaking Changes**: Existing packages continue to work

## Future Enhancements

Potential future improvements:
1. JSON Schema for manifest.json validation
2. Conversion tool to migrate YAML to JSON
3. Enhanced MCP tool integration
4. Support for Claude Desktop-specific features

## References

- [OpenHands Package POC](https://github.com/OpenHands/package-poc)
- [Claude Desktop Extensions](https://www.anthropic.com/engineering/desktop-extensions)
- [Model Context Protocol](https://modelcontextprotocol.io/)
- [Skill Packages Documentation](./skill-packages.md)
