from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# --------------------------------------------------
# 1. 데이터 폴더 설정
# --------------------------------------------------

CURRENT_DIR = Path(__file__).resolve().parent

# step2_retrieval -> src -> 7w -> 저장소 루트
SEVEN_W_DIR = CURRENT_DIR.parents[1]
REPO_ROOT = CURRENT_DIR.parents[2]

# PDF를 찾을 후보 폴더 (위에서부터 순서대로 확인)
CANDIDATE_DIRS = [
    REPO_ROOT / "6w" / "data",
    SEVEN_W_DIR / "data",
]

DATA_DIR = None
pdf_files = []

for folder in CANDIDATE_DIRS:
    files = sorted(folder.glob("공지*.pdf"))
    if files:
        DATA_DIR = folder
        pdf_files = files
        break

print("데이터 폴더:")
print(DATA_DIR)

# --------------------------------------------------
# 2. 공지 PDF 불러오기
# --------------------------------------------------

if not pdf_files:
    checked = "\n".join(str(f) for f in CANDIDATE_DIRS)
    raise FileNotFoundError(
        "공지 PDF를 찾을 수 없습니다. 아래 폴더를 확인했습니다:\n" + checked
    )

documents = []

for pdf_file in pdf_files:
    print(f"\n문서 불러오기: {pdf_file.name}")

    loader = PyPDFLoader(str(pdf_file))
    pages = loader.load()

    # 검색 결과에서 파일명을 확인할 수 있도록 추가
    for page in pages:
        page.metadata["file_name"] = pdf_file.name

    documents.extend(pages)

print("\n불러온 전체 페이지 수:", len(documents))

# --------------------------------------------------
# 3. Chunking
# --------------------------------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_documents(documents)

print("생성된 chunk 수:", len(chunks))

# --------------------------------------------------
# 4. Embedding 모델
# --------------------------------------------------

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# --------------------------------------------------
# 5. Chroma Vector Store 생성
# --------------------------------------------------

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    collection_name="week7_step2"
)

# --------------------------------------------------
# 6. Retriever 생성
# --------------------------------------------------

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

# --------------------------------------------------
# 7. 검색 테스트
# --------------------------------------------------

question = "인턴십 관련 공지의 주요 내용은 무엇인가요?"

print("\n질문:")
print(question)

docs = retriever.invoke(question)

# --------------------------------------------------
# 8. 검색 결과 출력
# --------------------------------------------------

for i, doc in enumerate(docs, start=1):
    print(f"\n========== 검색 결과 {i} ==========")
    print("파일:", doc.metadata.get("file_name", "확인 불가"))
    print("페이지:", doc.metadata.get("page", "확인 불가"))
    print("\n문서 내용:")
    print(doc.page_content[:800])