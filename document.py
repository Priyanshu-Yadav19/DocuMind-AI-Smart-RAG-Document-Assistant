import os
import time
import tempfile
import hashlib
import re
from typing import List, Optional

import streamlit as st
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:
    from langchain.text_splitter import RecursiveCharacterTextSplitter


# ===================== ENV =====================
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


# ===================== LIMITS / TUNING =====================
MAX_FILE_SIZE_MB = 30
MAX_TOTAL_SIZE_MB = 60

CHUNK_SIZE = 2000
CHUNK_OVERLAP = 120
MAX_TOTAL_CHUNKS = 1500

# Embedding batching
EMBED_BATCH_SIZE = 32
SLEEP_BETWEEN_EMBED_BATCHES = 1.0

# Retry/backoff
MAX_RETRIES = 6
BASE_WAIT_SEC = 10
MIN_LLM_WAIT_SEC = 60  # Gemini often tells ~59s; use 60s

# Retriever
RETRIEVER_K = 12


# ===================== UI =====================
st.set_page_config(page_title="Gemini RAG Chatbot", layout="wide")
st.title("📄 Gemini RAG Chatbot (RAG + Page + Chapter Logic, Safe)")
st.caption("PDF/TXT/DOCX • Semantic RAG + page/section ranges + chapter selection • Cached • Rate-limit safe")


# ===================== SESSION =====================
if "vectors" not in st.session_state:
    st.session_state.vectors = None
if "fingerprint" not in st.session_state:
    st.session_state.fingerprint = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_run_key" not in st.session_state:
    st.session_state.last_run_key = None
if "last_answer" not in st.session_state:
    st.session_state.last_answer = None
if "last_sources" not in st.session_state:
    st.session_state.last_sources = None


# ===================== MODELS =====================
llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0.2)

# ✅ Embeddings model your project supports
embeddings_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")


# ===================== PROMPT =====================
prompt = ChatPromptTemplate.from_template(
    """Answer the question using ONLY the provided context.
If the answer is not found in the context, say "I don't know."

<context>
{context}
</context>

Question: {input}
"""
)


# ===================== UPLOAD =====================
uploaded_files = st.file_uploader(
    "Upload PDF / TXT / DOCX files",
    type=["pdf", "txt", "docx"],
    accept_multiple_files=True,
)


# ===================== UTIL =====================
def validate_files(files) -> bool:
    total = 0.0
    for f in files:
        size = len(f.getvalue()) / (1024 * 1024)
        total += size
        if size > MAX_FILE_SIZE_MB:
            st.error(f"❌ {f.name} exceeds {MAX_FILE_SIZE_MB} MB")
            return False
    if total > MAX_TOTAL_SIZE_MB:
        st.error(f"❌ Total upload exceeds {MAX_TOTAL_SIZE_MB} MB")
        return False
    return True


def fingerprint_files(files) -> str:
    h = hashlib.sha256()
    for f in files:
        data = f.getvalue()
        h.update(f.name.encode("utf-8"))
        h.update(len(data).to_bytes(8, "big"))
        h.update(hashlib.sha256(data).digest())
    return h.hexdigest()


def cache_dir(fp: str) -> str:
    os.makedirs(".cache", exist_ok=True)
    return os.path.join(".cache", f"faiss_{fp}")


def is_rate_limit(e: Exception) -> bool:
    s = str(e).lower()
    return ("429" in s) or ("quota" in s) or ("rate" in s) or ("resourceexhausted" in s)


def backoff_sleep(attempt: int, min_wait: int = 0):
    wait = BASE_WAIT_SEC * (2 ** attempt)
    wait = max(wait, min_wait)
    time.sleep(wait)


# ===================== DOC LOADING =====================
def detect_chapter_number(text: str) -> Optional[int]:
    if not text:
        return None
    m = re.search(r"\bchapter\s+(\d+)\b", text.lower())
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def load_documents(files):
    docs = []
    tmp_paths = []

    try:
        for f in files:
            ext = f.name.split(".")[-1].lower()
            data = f.getvalue()

            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
                tmp.write(data)
                tmp_paths.append(tmp.name)
                path = tmp.name

            if ext == "pdf":
                loader = PyPDFLoader(path)
            elif ext == "txt":
                loader = TextLoader(path, encoding="utf-8")
            elif ext == "docx":
                loader = Docx2txtLoader(path)
            else:
                continue

            loaded = loader.load()
            for d in loaded:
                d.metadata["source"] = f.name
                d.metadata["ext"] = ext
                chap = detect_chapter_number(d.page_content[:4000])
                if chap is not None:
                    d.metadata["chapter"] = chap

            docs.extend(loaded)
    finally:
        for p in tmp_paths:
            try:
                os.remove(p)
            except OSError:
                pass

    return docs


def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.split_documents(docs)


# ===================== EMBEDDINGS + VECTORSTORE =====================
def embed_with_retry(texts: List[str]):
    vectors = []
    total = len(texts)
    for i in range(0, total, EMBED_BATCH_SIZE):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        for attempt in range(MAX_RETRIES):
            try:
                vecs = embeddings_model.embed_documents(batch)
                vectors.extend(vecs)
                break
            except Exception as e:
                if is_rate_limit(e):
                    st.warning(f"⏳ Embed rate limit hit. Retrying... (attempt {attempt+1}/{MAX_RETRIES})")
                    backoff_sleep(attempt)
                else:
                    raise
        time.sleep(SLEEP_BETWEEN_EMBED_BATCHES)
    return vectors


def load_cached(fp: str):
    index_dir = cache_dir(fp)
    if not os.path.exists(index_dir):
        return None
    try:
        return FAISS.load_local(index_dir, embeddings_model, allow_dangerous_deserialization=True)
    except Exception:
        return None


def save_cache(fp: str, vs: FAISS):
    vs.save_local(cache_dir(fp))


def build_vectorstore(files):
    docs = load_documents(files)
    chunks = split_docs(docs)

    if len(chunks) > MAX_TOTAL_CHUNKS:
        st.error(f"❌ Too many chunks ({len(chunks)}). Upload smaller docs.")
        st.stop()

    texts = [c.page_content for c in chunks]
    metas = [c.metadata for c in chunks]

    with st.spinner(f"Embedding {len(texts)} chunks..."):
        vectors = embed_with_retry(texts)

    return FAISS.from_embeddings(list(zip(texts, vectors)), embedding=embeddings_model, metadatas=metas)


# ===================== PAGE + CHAPTER REQUEST PARSERS =====================
def parse_page_request(q: str):
    s = q.lower()

    m = re.search(r"last\s+(\d+)\s+page", s)
    if m:
        return ("last", int(m.group(1)))

    m = re.search(r"page(?:s)?\s*(\d+)\s*(?:-|to)\s*(\d+)", s)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if a > b:
            a, b = b, a
        return ("range", a, b)

    return None


def parse_chapter_request(q: str) -> Optional[List[int]]:
    s = q.lower()

    # chapters 2-4
    m = re.search(r"chapters?\s+(\d+)\s*-\s*(\d+)", s)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        lo, hi = min(a, b), max(a, b)
        return list(range(lo, hi + 1))

    # chapters 2,3,4
    m = re.search(r"chapters?\s+((?:\d+\s*,\s*)*\d+)", s)
    if m:
        nums = [int(x.strip()) for x in m.group(1).split(",") if x.strip().isdigit()]
        return nums if nums else None

    # chapter 2
    m = re.search(r"\bchapter\s+(\d+)\b", s)
    if m:
        return [int(m.group(1))]

    return None


# ===================== SELECTION LOGIC =====================
def select_pdf_pages(all_docs, mode):
    pdf_docs = [d for d in all_docs if d.metadata.get("ext") == "pdf" and "page" in d.metadata]
    if not pdf_docs:
        return []

    sources = sorted(set(d.metadata.get("source", "Unknown") for d in pdf_docs))
    src0 = sources[0]
    pages = sorted([d for d in pdf_docs if d.metadata.get("source") == src0], key=lambda x: x.metadata.get("page", 0))

    if mode[0] == "last":
        n = mode[1]
        return pages[-n:] if n > 0 else []
    if mode[0] == "range":
        a, b = mode[1], mode[2]
        start = max(a - 1, 0)
        end = max(b - 1, 0)
        return pages[start : end + 1]
    return []


def select_sections_non_pdf(all_docs, mode):
    non_pdf = [d for d in all_docs if d.metadata.get("ext") in ("txt", "docx")]
    if not non_pdf:
        return []

    sources = sorted(set(d.metadata.get("source", "Unknown") for d in non_pdf))
    src0 = sources[0]
    src_docs = [d for d in non_pdf if d.metadata.get("source") == src0]

    chunks = split_docs(src_docs)
    if not chunks:
        return []

    if mode[0] == "last":
        n = mode[1]
        return chunks[-n:] if n > 0 else []
    if mode[0] == "range":
        a, b = mode[1], mode[2]
        start = max(a - 1, 0)
        end = max(b - 1, 0)
        return chunks[start : end + 1]
    return []


def select_chapters(all_docs, chapters: List[int]):
    return [d for d in all_docs if d.metadata.get("chapter") in set(chapters)]


# ===================== CONTEXT FORMAT =====================
def format_context(docs):
    out = []
    for d in docs:
        src = d.metadata.get("source", "Unknown")
        page = d.metadata.get("page", None)
        chap = d.metadata.get("chapter", None)

        head = f"[Source: {src}]"
        if chap is not None:
            head += f" [Chapter: {chap}]"
        if isinstance(page, int):
            head += f" [Page: {page+1}]"

        out.append(f"{head}\n{d.page_content}")
    return "\n\n".join(out)


# ===================== LLM SAFE INVOKE =====================
def llm_invoke_with_retry(llm_obj, messages):
    for attempt in range(MAX_RETRIES):
        try:
            return llm_obj.invoke(messages).content
        except Exception as e:
            if is_rate_limit(e):
                wait = max(MIN_LLM_WAIT_SEC, BASE_WAIT_SEC * (2 ** attempt))
                st.warning(f"⏳ LLM rate limit hit. Waiting {wait}s then retrying...")
                time.sleep(wait)
            else:
                raise
    st.error("LLM rate limit keeps happening. Try again in 1–2 minutes.")
    st.stop()


# ===================== QUESTION =====================
with st.form("qa"):
    question = st.text_input("Ask a question from your documents")
    ask = st.form_submit_button("Ask")


# ===================== RUN =====================
if ask and question:
    if not GOOGLE_API_KEY:
        st.error("❌ Missing GOOGLE_API_KEY in .env")
        st.stop()

    if not uploaded_files:
        st.warning("⚠️ Upload documents first")
        st.stop()

    if not validate_files(uploaded_files):
        st.stop()

    fp = fingerprint_files(uploaded_files)

    # Prevent accidental repeated LLM calls from Streamlit re-runs
    run_key = f"{fp}::{question.strip().lower()}"
    if st.session_state.last_run_key == run_key and st.session_state.last_answer is not None:
        st.info("✅ Same question already answered (avoiding extra API calls).")
        st.subheader("Answer")
        st.write(st.session_state.last_answer)
        with st.expander("🔍 Sources / Retrieved Chunks"):
            if st.session_state.last_sources:
                for d in st.session_state.last_sources:
                    src = d.metadata.get("source", "Unknown")
                    page = d.metadata.get("page", None)
                    chap = d.metadata.get("chapter", None)
                    label = f"**{src}**"
                    if chap is not None:
                        label += f" — Chapter {chap}"
                    if isinstance(page, int):
                        label += f" — Page {page+1}"
                    st.markdown(label)
                    st.write(d.page_content)
                    st.divider()
        st.stop()

    # Build/load vectorstore for semantic RAG
    if st.session_state.vectors is None or st.session_state.fingerprint != fp:
        cached = load_cached(fp)
        if cached:
            st.session_state.vectors = cached
            st.session_state.fingerprint = fp
            st.success("✅ Loaded cached embeddings")
        else:
            vs = build_vectorstore(uploaded_files)
            save_cache(fp, vs)
            st.session_state.vectors = vs
            st.session_state.fingerprint = fp
            st.success("✅ Embeddings created & cached")

    retriever = st.session_state.vectors.as_retriever(search_kwargs={"k": RETRIEVER_K})

    t0 = time.time()

    # -------- Decide routing (chapter > page > RAG) --------
    chapter_req = parse_chapter_request(question)
    page_req = parse_page_request(question)

    if chapter_req:
        all_docs = load_documents(uploaded_files)
        docs = select_chapters(all_docs, chapter_req)
        if not docs:
            st.warning("Couldn't match chapter headings. Try 'summarize pages X-Y' instead.")
    elif page_req:
        all_docs = load_documents(uploaded_files)
        docs = select_pdf_pages(all_docs, page_req)
        if not docs:
            docs = select_sections_non_pdf(all_docs, page_req)
        if not docs:
            st.warning("Couldn't select pages/sections (PDF might be scanned).")
    else:
        docs = retriever.get_relevant_documents(question)

    context = format_context(docs)
    messages = prompt.format_messages(context=context, input=question)

    answer = llm_invoke_with_retry(llm, messages)

    st.session_state.chat_history.append(HumanMessage(content=question))
    st.session_state.chat_history.append(AIMessage(content=answer))

    # Cache last run to avoid accidental repeated calls
    st.session_state.last_run_key = run_key
    st.session_state.last_answer = answer
    st.session_state.last_sources = docs

    st.subheader("Answer")
    st.write(answer)
    st.caption(f"⏱ {round(time.time() - t0, 2)} seconds")

    with st.expander("🔍 Sources / Retrieved Chunks"):
        if not docs:
            st.write("No context documents were selected/retrieved.")
        else:
            for d in docs:
                src = d.metadata.get("source", "Unknown")
                page = d.metadata.get("page", None)
                chap = d.metadata.get("chapter", None)

                label = f"**{src}**"
                if chap is not None:
                    label += f" — Chapter {chap}"
                if isinstance(page, int):
                    label += f" — Page {page+1}"

                st.markdown(label)
                st.write(d.page_content)
                st.divider()
