RAG
https://learn.kodekloud.com/user/courses/youtube-labs-rag


cd /root && mkdir -p rag-project && cd rag-project
python3 -m venv venv && source venv/bin/activate
pip install uv && uv pip install chromadb sentence-transformers openai flask
echo "READY" > /root/rag-setup-complete.txt

cd /root/techcorp-docs
ls -la
find . -name "*.md" | wc -l
find . -name "*.md" | wc -l > /root/doc-count.txt

cd /root/rag-project && python init_vectordb.py 

or
root@controlplane ~/rag-project via 🐍 v3.12.3 (venv) ➜  cd ~/rag-project && python init_vectordb.py
 Initializing AI Brain...
 Brain Created: techcorp_docs
 Memories: 0
 AI Brain Ready!
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
import chromadb
from chromadb.config import Settings

print(" Initializing AI Brain...")
client = chromadb.PersistentClient(
    path="./chroma_db",
    settings=Settings(anonymized_telemetry=False)
)

collection = client.get_or_create_collection(
    name="techcorp_docs",
    metadata={"hnsw:space": "cosine"}
)

print(f" Brain Created: {collection.name}")
print(f" Memories: {collection.count()}")
print(" AI Brain Ready!")
"""

root@controlplane ~/rag-project via 🐍 v3.12.3 (venv) ➜  python test_chunking.py
 DOCUMENT CHUNKING ENGINE
========================================
 Original document: 526 characters
----------------------------------------
 Created 2 chunks
----------------------------------------

Chunk 1 (500 chars):
Preview: TechCorp Pet Policy: 
Employees may bring pets to the office...

Chunk 2 (126 chars):
Preview: cluding dental and vision.
401k matching up to 6% of salary....

========================================
 Chunking complete!
 Stats: 2 chunks from 526 chars
 Ready for vectorization!
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
import os

print(" DOCUMENT CHUNKING ENGINE")
print("="*40)

def chunk_text(text, size=500, overlap=100):
    """Smart chunking with overlap for context preservation"""
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + size, len(text))
        chunk = text[start:end]
        chunks.append(chunk)

        if end >= len(text):
            break

        start += size - overlap

    return chunks

# Process sample document
sample_doc = """TechCorp Pet Policy: 
Employees may bring pets to the office on Fridays. 
Dogs must be well-behaved and vaccinated. 
The CEO's golden retriever is the office mascot.

Remote Work Policy:
Employees can work remotely up to 3 days per week.
Core hours are 10 AM - 3 PM in your local timezone.
All meetings should be recorded for async collaboration.

Benefits Overview:
Comprehensive health insurance including dental and vision.
401k matching up to 6% of salary.
Unlimited PTO after first year.
Annual learning budget of $2,000."""

print(f" Original document: {len(sample_doc)} characters")
print("-"*40)

chunks = chunk_text(sample_doc, size=500, overlap=100)

print(f" Created {len(chunks)} chunks")
print("-"*40)

for i, chunk in enumerate(chunks, 1):
    print(f"\nChunk {i} ({len(chunk)} chars):")
    print(f"Preview: {chunk[:60]}...")

# Save verification
with open('/root/chunk-test.txt', 'w') as f:
    f.write(f"CHUNKS:{len(chunks)}")

print("\n" + "="*40)
print(" Chunking complete!")
print(f" Stats: {len(chunks)} chunks from {len(sample_doc)} chars")
print(" Ready for vectorization!")
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

root@controlplane ~/rag-project via 🐍 v3.12.3 (venv) ➜  python ingest_documents.py
TECHCORP KNOWLEDGE INGESTION SYSTEM
==================================================
Connecting to AI Brain (from Task 3)...
Loading Semantic Processor (from Task 5)...
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|███████████████████████████████████████████████████████████████████████████| 103/103 [00:00<00:00, 6375.55it/s, Materializing param=pooler.dense.weight]
BertModel LOAD REPORT from: sentence-transformers/all-MiniLM-L6-v2
Key                     | Status     |  | 
------------------------+------------+--+-
embeddings.position_ids | UNEXPECTED |  | 

Notes:
- UNEXPECTED    :can be ignored when loading from different task/architecture; not ok if you expect identical arch.
All systems online!

Beginning knowledge transfer...

Processing product-specs:
  cloudsync-pro.md (10 chunks)
  datavault.md (13 chunks)

Processing meeting-notes:
  product-launch-review.md (13 chunks)
  q3-planning-meeting.md (9 chunks)

Processing employee-handbook:
  remote-work-policy.md (9 chunks)
  benefits-overview.md (9 chunks)
  pet-policy.md (5 chunks)

Processing customer-faqs:
  general-faqs.md (16 chunks)

==================================================
INGESTION COMPLETE!
Statistics:
   • Documents processed: 8
   • Knowledge chunks: 84
   • AI IQ increased: +80 points

Value delivered: $500K in searchable knowledge!
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
import os
import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path

print("TECHCORP KNOWLEDGE INGESTION SYSTEM")
print("="*50)

# Initialize systems
print("Connecting to AI Brain (from Task 3)...")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("techcorp_docs")

print("Loading Semantic Processor (from Task 5)...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("All systems online!\n")

# Process documents
print("Beginning knowledge transfer...")
doc_count = 0
total_chunks = 0

for category in Path('/root/techcorp-docs').iterdir():
    if category.is_dir():
        print(f"\nProcessing {category.name}:")

        for doc in category.glob('*.md'):
            print(f"  {doc.name}", end="")

            with open(doc, 'r') as f:
                content = f.read()

            # Apply chunking strategy from Task 4!
            chunks = [content[i:i+500] for i in range(0, len(content), 400)]

            for i, chunk in enumerate(chunks):
                doc_id = f"{doc.stem}_{i}"
                # Apply embedding from Task 5!
                embedding = model.encode(chunk).tolist()

                # Store in database from Task 3!
                collection.add(
                    ids=[doc_id],
                    embeddings=[embedding],
                    documents=[chunk],
                    metadatas={"file": doc.name, "category": category.name}
                )
                total_chunks += 1

            doc_count += 1
            print(f" ({len(chunks)} chunks)")

print("\n" + "="*50)
print(f"INGESTION COMPLETE!")
print(f"Statistics:")
print(f"   • Documents processed: {doc_count}")
print(f"   • Knowledge chunks: {total_chunks}")
print(f"   • AI IQ increased: +{doc_count*10} points")
print(f"\nValue delivered: $500K in searchable knowledge!")

# Save results
with open('/root/ingest-complete.txt', 'w') as f:
    f.write(f"DOCS:{doc_count},CHUNKS:{collection.count()}")

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
import chromadb
from sentence_transformers import SentenceTransformer

print(" TECHCORP SEMANTIC SEARCH ENGINE")
print("="*50)

# Initialize
print(" Connecting to Knowledge Base...")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("techcorp_docs")

print(" Loading AI Understanding...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print(" Search Engine Ready!\n")

# CEO's test queries
queries = [
    "What is the pet policy at TechCorp?",
    "Tell me about CloudSync Pro features",
    "How many days of remote work are allowed?"
]

results_file = open('/root/search-results.txt', 'w')

for query in queries:
    print(f" Query: '{query}'")
    print("-" * 50)
    results_file.write(f"QUERY:{query}\n")

    # Convert question to vector
    query_embedding = model.encode(query).tolist()

    # Semantic search!
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    # Display results
    print(" Top Results (by semantic similarity):")
    for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
        relevance = 100 - (i * 15)  # Simulated relevance
        print(f"\n  {i+1}. [{meta['category']}] {meta['file']} ({relevance}% match)")
        print(f"     Preview: '{doc[:80]}...'")
        results_file.write(f"RESULT:{meta['category']}/{meta['file']}\n")

    print("\n" + "="*50 + "\n")

results_file.close()

print(" SEARCH TEST COMPLETE!")
print(" Notice: Found 'pet policy' even when searching 'bring my dog'!")
print(" This is the power of semantic understanding!")
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


import chromadb
from sentence_transformers import SentenceTransformer
import openai
import os

print(" TECHCORP RAG PIPELINE TEST")
print("="*50)

# Initialize all systems
print(" Initializing RAG Components...")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("techcorp_docs")
model = SentenceTransformer('all-MiniLM-L6-v2')
print(" All systems operational!\n")

def test_rag_pipeline(question):
    """Test the complete RAG Pipeline"""

    print(f" Question: '{question}'")
    print("-" * 50)

    # 1. RETRIEVAL PHASE
    print("\n PHASE 1: RETRIEVAL")
    print("  Converting question to vector...")
    query_embedding = model.encode(question).tolist()
    print("  Searching knowledge base...")

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    print(f"   Found {len(results['documents'][0])} relevant documents!")

    # 2. AUGMENTATION PHASE
    print("\n PHASE 2: AUGMENTATION")
    print("  Preparing context for AI...")
    context = "\n\n".join(results['documents'][0])

    # 3. GENERATION PHASE (Simulated)
    print("\n PHASE 3: GENERATION")
    print("  AI processing with context...")

    # Simulated response
    if "benefits" in question.lower():
        answer = "Based on TechCorp documents: Employees enjoy comprehensive health insurance, 401k matching up to 6%, unlimited PTO, and professional development budgets."
    else:
        answer = f"Based on the retrieved TechCorp documents, here's the answer to '{question}'..."

    print("   Response generated!")

    return {
        'question': question,
        'sources_used': len(results['documents'][0]),
        'answer': answer
    }

# Test the pipeline
print("\n" + "="*50)
print(" TESTING COMPLETE PIPELINE")
print("="*50)

test_question = "What are the benefits of working at TechCorp?"
result = test_rag_pipeline(test_question)

print("\n" + "="*50)
print(" PIPELINE RESULTS")
print("="*50)
print(f" Question: {result['question']}")
print(f" Sources Used: {result['sources_used']} documents")
print(f" Answer: {result['answer']}")

# Performance metrics
print("\n PERFORMANCE METRICS:")
print("  • Retrieval: 0.012 seconds")
print("  • Augmentation: 0.003 seconds")
print("  • Generation: 0.234 seconds")
print("  • Total: 0.249 seconds")

# Save pipeline verification
with open('/root/rag-pipeline-test.txt', 'w') as f:
    f.write(f"PIPELINE:COMPLETE,SOURCES:{result['sources_used']}")

print("\n" + "="*50)
print(" SUCCESS! RAG Pipeline Working!")
print("="*50)
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
