# RAG Content Update - Changelog

## Summary

Updated the backend to use RAG-optimized content with resolved AsciiDoc attributes instead of raw .adoc files. This ensures the AI assistant sees the same content that end users see, with all variables like `{lab_name}`, `{ssh_user}`, `{guid}` properly substituted.

## Changes Made

### 1. New Antora Extension (`content/lib/rag-export.js`)

Created an Antora extension that exports content with resolved attributes:
- Hooks into the `documentsConverted` event (after attribute substitution, before HTML rendering)
- Reads original `.adoc` files and substitutes all attributes from `antora.yml`
- Exports to `./rag-content/` directory with metadata headers
- Each file includes JSON metadata (title, component, module, URL, attributes)

### 2. Updated Dockerfile (`Dockerfile.backend`)

**Before:**
```dockerfile
COPY content/ /app/content/
ENV CONTENT_DIR=/app/content
ENV PDF_DIR=/app/content/modules/ROOT/assets/techdocs
```

**After:**
```dockerfile
# Copy RAG-optimized content (with resolved attributes)
COPY rag-content/ /app/rag-content/

# Copy techdocs PDFs (keep original structure for PDF_DIR)
COPY content/modules/ROOT/assets/techdocs/ /app/content/modules/ROOT/assets/techdocs/

# Use RAG-optimized content with resolved AsciiDoc attributes
ENV CONTENT_DIR=/app/rag-content
# Keep using techdocs from original content location
ENV PDF_DIR=/app/content/modules/ROOT/assets/techdocs
```

### 3. Updated RAG Initialization (`backend/rag_init.py`)

- Added JSON import for metadata parsing
- Updated `_load_asciidoc_content()` to:
  - Detect and load new RAG content format (`.txt` files with metadata)
  - Parse metadata headers to extract title, module, etc.
  - Fall back to raw `.adoc` files if RAG content not available
  - Support both formats for backward compatibility

### 4. Updated Application Defaults (`backend/app.py`)

Changed default content directory from `../content` to `../rag-content` for local development.

### 5. Updated Build Configuration (`default-site.yml`)

Enabled the RAG export extension:
```yaml
antora:
  extensions:
    - require: ./content/lib/rag-export.js
      enabled: true
      outputDir: ./rag-content
```

### 6. Updated .gitignore

Added `rag-content/` to ignore list since it's generated content.

### 7. Documentation Updates

- Added `content/lib/RAG-EXPORT-README.md` - Technical documentation for the extension
- Updated `README.adoc` - Added "Building RAG-Optimized Content" section

## Verification

Tested attribute substitutions in exported content:
- ✅ `{lab_name}` → "I'm the lab_name var"
- ✅ `{my_var}` → "foo"
- ✅ `{ssh_user}` → "devops"
- ✅ `{guid}` → "my-guid"
- ✅ All attributes from `content/antora.yml` properly resolved

## Build Process

### Before Deployment

```bash
# Generate RAG-optimized content
antora default-site.yml

# This creates:
# - www/ - The workshop HTML site
# - rag-content/ - Processed content for RAG
```

### Docker Build

The `rag-content/` directory must exist before building the Docker image:

```bash
# 1. Build Antora site (generates rag-content/)
antora default-site.yml

# 2. Build Docker image
podman build -f Dockerfile.backend -t ai-assistant-backend .
```

## Benefits

1. **Accurate RAG**: Users can query with actual values (e.g., "devops user") not variables (`{ssh_user}`)
2. **Single Source of Truth**: Same attribute values in workshop docs and AI responses
3. **Automatic Updates**: Rebuilding Antora updates both docs and RAG content
4. **Better Citations**: Structured metadata improves source attribution
5. **Backward Compatible**: Falls back to raw `.adoc` files if RAG content not available

## Migration Notes

### For Existing Deployments

1. Run `antora default-site.yml` before building Docker images
2. The deploy script (`deploy-backend.sh`) already runs Antora, so no changes needed there
3. For manual Docker builds, ensure you run Antora first

### For Local Development

The default content path has changed to `../rag-content`, but the code falls back to raw `.adoc` files if not available. To use the new format locally:

```bash
# From project root
antora default-site.yml

# Then run backend
cd backend
python app.py
```

## Technical Details

See `content/lib/RAG-EXPORT-README.md` for:
- How the Antora extension works
- Antora build pipeline details
- Output file format specification
- Configuration options
- Troubleshooting guide
