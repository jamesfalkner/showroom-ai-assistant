# RAG Export Extension for Antora

## Problem Statement

When using raw AsciiDoc files for RAG (Retrieval Augmented Generation), the content doesn't match what end-users see because AsciiDoc variables/attributes (like `{lab_name}`, `{ssh_user}`, `{guid}`) are not substituted until the build process.

## Solution

This Antora extension (`rag-export.js`) intercepts the Antora build process **after attribute substitution but before HTML rendering**, giving you access to the "user-facing" content with all variables resolved.

## How It Works

### Antora Build Pipeline

```
1. Load & Parse .adoc files
2. Resolve Attributes ({lab_name} → "I'm the lab_name var")  ← We intercept here!
3. Convert to HTML
4. Apply UI theme
```

### The Extension

The extension hooks into the `documentsConverted` event, which fires after step 2. It then:

1. Reads the original `.adoc` source files
2. Substitutes all AsciiDoc attributes defined in `antora.yml`
3. Exports to `./rag-content/` directory with metadata

### Output Format

Each exported file contains:
- **Metadata** (JSON): Title, component, module, URL, relevant attributes
- **Content**: AsciiDoc source with all `{variables}` substituted

Example output (`rag-content/modules-ROOT-index.txt`):
```
---
METADATA:
{
  "title": "I'm the lab_name var",
  "component": "modules",
  "version": "master",
  "module": "ROOT",
  "originalPath": "index.adoc",
  "url": "/modules/index.html",
  "relevantAttributes": {
    "lab_name": "I'm the lab_name var",
    "guid": "my-guid",
    "ssh_user": "devops",
    ...
  }
}
---

= I'm the lab_name var

The terminal window is logged in as the `devops` user...
```

## Usage

### 1. Enable the Extension

Already enabled in `default-site.yml`:
```yaml
antora:
  extensions:
    - require: ./content/lib/rag-export.js
      enabled: true
      outputDir: ./rag-content  # Optional: customize output directory
```

### 2. Run Antora Build

```bash
antora default-site.yml
```

This will:
- Build your normal site to `./www/`
- Export RAG-friendly content to `./rag-content/`

### 3. Use Exported Content for RAG

Point your RAG ingestion to `./rag-content/` instead of the raw `.adoc` files.

## Configuration Options

In `default-site.yml`:
```yaml
- require: ./content/lib/rag-export.js
  enabled: true           # Set to false to disable export
  outputDir: ./rag-content  # Customize output directory
```

## Updating RAG Ingestion

To use the exported content, update `backend/rag_init.py`:

### Option 1: Use exported files directly

```python
def _load_asciidoc_content(content_path: Path, min_chunk_size: int) -> List[dict]:
    """Load exported RAG content with resolved attributes"""
    documents = []
    rag_content_dir = Path("./rag-content")  # Path to exported content

    if not rag_content_dir.exists():
        logger.warning(f"RAG content directory not found: {rag_content_dir}")
        return documents

    for exported_file in rag_content_dir.glob("*.txt"):
        try:
            content = exported_file.read_text(encoding='utf-8')

            # Parse metadata and content
            if content.startswith('---\nMETADATA:\n'):
                parts = content.split('---\n', 2)
                metadata_json = parts[1].replace('METADATA:\n', '')
                metadata = json.loads(metadata_json)
                actual_content = parts[2] if len(parts) > 2 else ''

                if len(actual_content.strip()) > min_chunk_size:
                    documents.append({
                        'title': metadata['title'],
                        'content': actual_content,
                        'file_path': metadata['originalPath'],
                        'module': f"{metadata['component']} - {metadata['module']}"
                    })
        except Exception as e:
            logger.warning(f"Error loading {exported_file}: {e}")

    return documents
```

### Option 2: Regenerate on RAG init

Add to `backend/rag_init.py`:
```python
import subprocess

async def initialize_vector_store(client, content_dir: str, ...):
    # First, regenerate RAG content with attribute substitution
    logger.info("Regenerating content with resolved attributes...")
    subprocess.run(["antora", "default-site.yml"], check=True)

    # Then load from rag-content directory
    ...
```

## Benefits

1. **Accurate RAG**: Users query with actual values (e.g., "devops user") not variables (`{ssh_user}`)
2. **Single Source of Truth**: Same attribute values in docs and RAG
3. **Automatic Updates**: Rebuild Antora to update both docs and RAG content
4. **Metadata Included**: Each file has structured metadata for better retrieval

## Verified Substitutions

The following attribute substitutions have been verified:
- ✅ `{lab_name}` → "I'm the lab_name var"
- ✅ `{my_var}` → "foo"
- ✅ `{ssh_user}` → "devops"
- ✅ `{guid}` → "my-guid"
- ✅ `{ssh_password}` → "devops"
- ✅ All other attributes in `content/antora.yml`

## Adding More Attributes

To export additional attributes, edit `content/lib/rag-export.js`:

```javascript
function extractRelevantAttributes(attributes) {
  const relevantAttrs = [
    'lab_name', 'guid', 'ssh_user', 'ssh_password', 'ssh_command',
    'release-version', 'workshop_title', 'assistant_name',
    'my_var', 'welcome_message',
    'your_custom_attribute'  // Add your attributes here
  ]
  ...
}
```

## Troubleshooting

### Files not being exported
Check that the extension is enabled in `default-site.yml` and that you're running the full Antora build.

### Attributes not substituted
Make sure attributes are defined in `content/antora.yml` under `asciidoc.attributes`.

### Path errors
The extension assumes standard Antora structure: `content/modules/ROOT/pages/*.adoc`

## Architecture Note

This solution uses **Antora extension events**, not Asciidoctor processor hooks. The key advantage is that we get access to the fully-resolved attribute context from Antora's component descriptors.
