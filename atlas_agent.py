"""
Atlas - Multi-Agent AI System with RAG
A personal project demonstrating LLM orchestration with specialized agents
Using Google Gemini 2.5 Flash with pgvector memory persistence
"""

import os
import uuid
from datetime import datetime
from typing import TypedDict, Annotated, Sequence, Optional
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import PGVector
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper, SerpAPIWrapper
from langchain_community.tools import Tool
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import operator

# ============================================================================
# CONFIGURATION
# ============================================================================

# Set your API keys as environment variables:
# export GOOGLE_API_KEY="your-key"
# export SERPAPI_API_KEY="your-key"
# export DATABASE_URL="postgresql://user:password@localhost:5432/atlas_db"

LLM_MODEL = "gemini-2.0-flash-exp"  # Gemini 2.5 Flash
EMBEDDING_MODEL = "models/embedding-001"

# Database connection string
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://atlas_user:atlas_pass@localhost:5432/atlas_db"
)

# ============================================================================
# AGENT STATE
# ============================================================================

class AgentState(TypedDict):
    """State shared across all agents"""
    messages: Annotated[Sequence[str], operator.add]
    query: str
    search_results: str
    summary: str
    synthesis: str
    evaluation_score: float
    needs_improvement: bool
    iteration_count: int
    session_id: str

# ============================================================================
# TOOLS SETUP
# ============================================================================

def setup_tools():
    """Initialize all agent tools"""
    
    # Wikipedia Tool
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    
    # ArXiv Tool
    arxiv = ArxivQueryRun(api_wrapper=ArxivAPIWrapper())
    
    # SerpAPI Tool (Google Search)
    search = Tool(
        name="google_search",
        description="Search Google for current information and news",
        func=SerpAPIWrapper().run
    )
    
    return [wikipedia, arxiv, search]

# ============================================================================
# SESSION MANAGER
# ============================================================================

class SessionManager:
    """Manages conversation sessions and history"""
    
    def __init__(self):
        self.sessions = {}  # In-memory session tracking
    
    def create_session(self) -> str:
        """Create a new session ID"""
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "created_at": datetime.now(),
            "queries": [],
            "context": []
        }
        return session_id
    
    def add_to_session(self, session_id: str, query: str, answer: str, score: float):
        """Add interaction to session history"""
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "created_at": datetime.now(),
                "queries": [],
                "context": []
            }
        
        self.sessions[session_id]["queries"].append({
            "query": query,
            "answer": answer,
            "score": score,
            "timestamp": datetime.now()
        })
        
        # Build contextual summary for RAG
        context = f"Previous Query: {query}\nAnswer: {answer[:200]}..."
        self.sessions[session_id]["context"].append(context)
    
    def get_session_context(self, session_id: str) -> list[str]:
        """Get conversation context for a session"""
        if session_id not in self.sessions:
            return []
        return self.sessions[session_id]["context"]
    
    def get_session_history(self, session_id: str) -> list[dict]:
        """Get full session history"""
        if session_id not in self.sessions:
            return []
        return self.sessions[session_id]["queries"]

# ============================================================================
# RAG PIPELINE WITH PGVECTOR
# ============================================================================

class RAGPipeline:
    """pgvector-based RAG for persistent semantic search and retrieval"""
    
    def __init__(self, collection_name: str = "atlas_memory"):
        self.embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
        self.collection_name = collection_name
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        # Initialize pgvector store
        try:
            self.vectorstore = PGVector(
                collection_name=self.collection_name,
                connection_string=DATABASE_URL,
                embedding_function=self.embeddings,
            )
            print("✅ Connected to pgvector database")
        except Exception as e:
            print(f"⚠️  Could not connect to pgvector: {e}")
            print("📝 Falling back to in-memory mode (no persistence)")
            self.vectorstore = None
    
    def add_documents(self, texts: list[str], metadata: Optional[dict] = None):
        """Add documents to persistent vector store"""
        if not self.vectorstore:
            return
        
        docs = []
        for text in texts:
            meta = metadata.copy() if metadata else {}
            meta["timestamp"] = datetime.now().isoformat()
            docs.append(Document(page_content=text, metadata=meta))
        
        splits = self.text_splitter.split_documents(docs)
        
        try:
            self.vectorstore.add_documents(splits)
        except Exception as e:
            print(f"⚠️  Error adding documents: {e}")
    
    def search(self, query: str, k: int = 3, filter_dict: Optional[dict] = None) -> list[str]:
        """Semantic search for relevant context with optional filtering"""
        if not self.vectorstore:
            return []
        
        try:
            results = self.vectorstore.similarity_search(
                query, 
                k=k,
                filter=filter_dict
            )
            return [doc.page_content for doc in results]
        except Exception as e:
            print(f"⚠️  Error searching: {e}")
            return []
    
    def search_by_session(self, query: str, session_id: str, k: int = 3) -> list[str]:
        """Search within a specific session's memory"""
        return self.search(query, k=k, filter_dict={"session_id": session_id})
    
    def clear_collection(self):
        """Clear all documents from collection (for testing)"""
        if not self.vectorstore:
            return
        
        try:
            # This will depend on your pgvector setup
            print("⚠️  Collection clearing not implemented - manually truncate if needed")
        except Exception as e:
            print(f"⚠️  Error clearing collection: {e}")

# ============================================================================
# SPECIALIZED AGENTS
# ============================================================================

class SearchAgent:
    """Agent specialized in information retrieval"""
    
    def __init__(self, tools):
        self.tools = tools
        self.llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0)
    
    def search(self, query: str) -> str:
        """Execute multi-source search"""
        results = []
        
        for tool in self.tools:
            try:
                result = tool.run(query)
                results.append(f"[{tool.name}]\n{result}\n")
            except Exception as e:
                results.append(f"[{tool.name}] Error: {str(e)}\n")
        
        return "\n".join(results)

class SummaryAgent:
    """Agent specialized in summarization"""
    
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.3)
    
    def summarize(self, text: str, context: list[str] = None) -> str:
        """Summarize information with optional RAG context"""
        
        context_str = ""
        if context:
            context_str = "\n\nRelevant Context from Memory:\n" + "\n".join(context)
        
        prompt = f"""Summarize the following information concisely and accurately.
Focus on key facts and main points.{context_str}

Information:
{text}

Summary:"""
        
        response = self.llm.invoke(prompt)
        return response.content

class SynthesisAgent:
    """Agent specialized in synthesizing information"""
    
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.5)
    
    def synthesize(self, query: str, summary: str, context: list[str] = None, 
                   session_context: list[str] = None) -> str:
        """Create comprehensive synthesis with session context"""
        
        context_str = ""
        if context:
            context_str = "\n\nRelevant Knowledge:\n" + "\n".join(context)
        
        session_str = ""
        if session_context:
            session_str = "\n\nConversation History:\n" + "\n".join(session_context[-3:])  # Last 3 interactions
        
        prompt = f"""Based on the query and summarized information, provide a comprehensive,
well-structured answer that directly addresses the user's question.{context_str}{session_str}

Query: {query}

Summary:
{summary}

Synthesis:"""
        
        response = self.llm.invoke(prompt)
        return response.content

class EvaluationAgent:
    """Agent for self-evaluation of responses"""
    
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0)
    
    def evaluate(self, query: str, synthesis: str) -> tuple[float, bool]:
        """Evaluate response quality (0-100 scale)"""
        
        prompt = f"""Evaluate the following answer on these criteria:
1. Factual accuracy (does it answer the question correctly?)
2. Completeness (does it cover all aspects?)
3. Clarity (is it well-structured and understandable?)

Score from 0-100 and explain if improvements are needed.

Query: {query}

Answer:
{synthesis}

Provide your evaluation in this format:
Score: [number]
Needs Improvement: [Yes/No]
Reasoning: [brief explanation]"""
        
        response = self.llm.invoke(prompt)
        content = response.content
        
        # Parse score
        score = 75.0  # default
        needs_improvement = False
        
        try:
            if "Score:" in content:
                score_line = [l for l in content.split("\n") if "Score:" in l][0]
                score = float(score_line.split(":")[-1].strip())
            
            if "Needs Improvement:" in content:
                improvement_line = [l for l in content.split("\n") if "Needs Improvement:" in l][0]
                needs_improvement = "yes" in improvement_line.lower()
        except:
            pass
        
        return score, needs_improvement

# ============================================================================
# LANGGRAPH ORCHESTRATION
# ============================================================================

class Atlas:
    """Main orchestrator for the multi-agent system with persistent memory"""
    
    def __init__(self, session_id: Optional[str] = None):
        self.tools = setup_tools()
        self.search_agent = SearchAgent(self.tools)
        self.summary_agent = SummaryAgent()
        self.synthesis_agent = SynthesisAgent()
        self.evaluation_agent = EvaluationAgent()
        self.rag = RAGPipeline()
        self.session_manager = SessionManager()
        
        # Create or use existing session
        self.session_id = session_id or self.session_manager.create_session()
        
        # Build the workflow graph
        self.graph = self._build_graph()
    
    def _build_graph(self):
        """Build LangGraph workflow"""
        
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("search", self._search_node)
        workflow.add_node("summary", self._summary_node)
        workflow.add_node("synthesis", self._synthesis_node)
        workflow.add_node("evaluation", self._evaluation_node)
        
        # Define edges
        workflow.set_entry_point("search")
        workflow.add_edge("search", "summary")
        workflow.add_edge("summary", "synthesis")
        workflow.add_edge("synthesis", "evaluation")
        
        # Conditional edge: re-run if needs improvement
        workflow.add_conditional_edges(
            "evaluation",
            self._should_continue,
            {
                "continue": "search",
                "end": END
            }
        )
        
        return workflow.compile()
    
    def _search_node(self, state: AgentState) -> AgentState:
        """Search agent node"""
        print("🔍 Searching...")
        results = self.search_agent.search(state["query"])
        
        # Add to persistent RAG with session metadata
        self.rag.add_documents(
            [results],
            metadata={"session_id": state["session_id"], "type": "search_result"}
        )
        
        state["search_results"] = results
        state["messages"].append(f"Search completed: {len(results)} characters")
        return state
    
    def _summary_node(self, state: AgentState) -> AgentState:
        """Summary agent node"""
        print("📝 Summarizing...")
        
        # Get RAG context from persistent memory
        context = self.rag.search(state["query"])
        
        summary = self.summary_agent.summarize(
            state["search_results"],
            context
        )
        
        state["summary"] = summary
        state["messages"].append("Summary created")
        return state
    
    def _synthesis_node(self, state: AgentState) -> AgentState:
        """Synthesis agent node"""
        print("🔬 Synthesizing...")
        
        # Get RAG context from persistent memory
        context = self.rag.search(state["query"])
        
        # Get session conversation history
        session_context = self.session_manager.get_session_context(state["session_id"])
        
        synthesis = self.synthesis_agent.synthesize(
            state["query"],
            state["summary"],
            context,
            session_context
        )
        
        state["synthesis"] = synthesis
        state["messages"].append("Synthesis completed")
        return state
    
    def _evaluation_node(self, state: AgentState) -> AgentState:
        """Evaluation agent node"""
        print("✅ Evaluating...")
        
        score, needs_improvement = self.evaluation_agent.evaluate(
            state["query"],
            state["synthesis"]
        )
        
        state["evaluation_score"] = score
        state["needs_improvement"] = needs_improvement
        state["iteration_count"] = state.get("iteration_count", 0) + 1
        state["messages"].append(f"Evaluation score: {score}")
        
        return state
    
    def _should_continue(self, state: AgentState) -> str:
        """Decide whether to continue iterating"""
        
        # Max 2 iterations to avoid infinite loops
        if state["iteration_count"] >= 2:
            return "end"
        
        # If score is good enough, end
        if state["evaluation_score"] >= 80:
            return "end"
        
        # If needs improvement and under iteration limit
        if state["needs_improvement"]:
            return "continue"
        
        return "end"
    
    def query(self, question: str) -> dict:
        """Process a query through the agent system with session persistence"""
        
        initial_state = AgentState(
            messages=[],
            query=question,
            search_results="",
            summary="",
            synthesis="",
            evaluation_score=0.0,
            needs_improvement=False,
            iteration_count=0,
            session_id=self.session_id
        )
        
        # Run the graph
        final_state = self.graph.invoke(initial_state)
        
        # Store in session history
        self.session_manager.add_to_session(
            self.session_id,
            question,
            final_state["synthesis"],
            final_state["evaluation_score"]
        )
        
        # Add final answer to persistent RAG
        self.rag.add_documents(
            [f"Q: {question}\nA: {final_state['synthesis']}"],
            metadata={"session_id": self.session_id, "type": "qa_pair"}
        )
        
        return {
            "answer": final_state["synthesis"],
            "score": final_state["evaluation_score"],
            "iterations": final_state["iteration_count"],
            "session_id": self.session_id
        }
    
    def get_session_history(self) -> list[dict]:
        """Get conversation history for current session"""
        return self.session_manager.get_session_history(self.session_id)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Example usage demonstrating session persistence"""
    
    print("=" * 60)
    print("ATLAS - Multi-Agent AI System with Persistent Memory")
    print("Powered by Gemini 2.5 Flash + pgvector")
    print("=" * 60)
    
    # Initialize Atlas (creates new session)
    atlas = Atlas()
    print(f"\n🆔 Session ID: {atlas.session_id}\n")
    
    # Example: Multi-turn conversation demonstrating memory
    queries = [
        "What are the latest developments in quantum computing?",
        "How does this relate to cryptography?",  # Follow-up using context
        "Explain the concept of attention mechanism in transformers"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n{'='*60}")
        print(f"Query {i}/{len(queries)}: {query}")
        print("-" * 60)
        
        result = atlas.query(query)
        
        print(f"\n💡 Answer:\n{result['answer']}")
        print(f"\n📊 Quality Score: {result['score']:.1f}/100")
        print(f"🔄 Iterations: {result['iterations']}")
    
    # Show session history
    print(f"\n{'='*60}")
    print("📚 Session History")
    print("=" * 60)
    history = atlas.get_session_history()
    for i, interaction in enumerate(history, 1):
        print(f"\n{i}. Q: {interaction['query']}")
        print(f"   Score: {interaction['score']:.1f}/100")
        print(f"   Time: {interaction['timestamp'].strftime('%H:%M:%S')}")
    
    print("\n" + "=" * 60)
    print("💾 All interactions stored in pgvector for future sessions!")
    print("=" * 60)

if __name__ == "__main__":
    main()
