"""
Atlas - Multi-Agent AI System with RAG
A personal project demonstrating LLM orchestration with specialized agents
Using Google Gemini 2.5 Flash
"""

import os
from typing import TypedDict, Annotated, Sequence
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper, SerpAPIWrapper
from langchain_community.tools import Tool
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import operator

# ============================================================================
# CONFIG
# ============================================================================

LLM_MODEL = "gemini-2.5-flash-lite"
EMBEDDING_MODEL = "models/embedding-001"

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
# RAG w FAISS
# ============================================================================

class RAGPipeline:
    """FAISS-based RAG for semantic search and retrieval"""
    
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
        self.vectorstore = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
    
    def add_documents(self, texts: list[str]):
        """Add documents to vector store"""
        docs = [Document(page_content=text) for text in texts]
        splits = self.text_splitter.split_documents(docs)
        
        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents(splits, self.embeddings)
        else:
            self.vectorstore.add_documents(splits)
    
    def search(self, query: str, k: int = 3) -> list[str]:
        """Semantic search for relevant context"""
        if self.vectorstore is None:
            return []
        
        results = self.vectorstore.similarity_search(query, k=k)
        return [doc.page_content for doc in results]

# ============================================================================
# AGENTS
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
            context_str = "\n\nRelevant Context:\n" + "\n".join(context)
        
        prompt = f"""Summarize the given information concisely and accurately. Focus on key facts and main points. {context_str}

                    Information:
                    {text}
                    
                    Summary:"""
        
        response = self.llm.invoke(prompt)
        return response.content

class SynthesisAgent:
    """Agent specialized in synthesizing information"""
    
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.5)
    
    def synthesize(self, query: str, summary: str, context: list[str] = None) -> str:
        """Create comprehensive synthesis"""
        
        context_str = ""
        if context:
            context_str = "\n\nAdditional Context:\n" + "\n".join(context)
        
        prompt = f"""Based on the query and summarized information, provide a comprehensive,
        well-structured answer that directly addresses the user's question.{context_str}

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
    """Main orchestrator for the multi-agent system"""
    
    def __init__(self):
        self.tools = setup_tools()
        self.search_agent = SearchAgent(self.tools)
        self.summary_agent = SummaryAgent()
        self.synthesis_agent = SynthesisAgent()
        self.evaluation_agent = EvaluationAgent()
        self.rag = RAGPipeline()
        
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
        
        # Add to RAG
        self.rag.add_documents([results])
        
        state["search_results"] = results
        state["messages"].append(f"Search completed: {len(results)} characters")
        return state
    
    def _summary_node(self, state: AgentState) -> AgentState:
        """Summary agent node"""
        print("📝 Summarizing...")
        
        # Get RAG context
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
        
        # Get RAG context
        context = self.rag.search(state["query"])
        
        synthesis = self.synthesis_agent.synthesize(
            state["query"],
            state["summary"],
            context
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
        """Process a query through the agent system"""
        
        initial_state = AgentState(
            messages=[],
            query=question,
            search_results="",
            summary="",
            synthesis="",
            evaluation_score=0.0,
            needs_improvement=False,
            iteration_count=0
        )
        
        # Run the graph
        final_state = self.graph.invoke(initial_state)
        
        return {
            "answer": final_state["synthesis"],
            "score": final_state["evaluation_score"],
            "iterations": final_state["iteration_count"]
        }

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Example usage"""
    
    print("=" * 60)
    print("ATLAS - Multi-Agent AI System (Powered by Gemini 2.5 Flash)")
    print("=" * 60)
    
    # Initialize Atlas
    atlas = Atlas()
    
    # Example queries
    queries = [
        "What are the latest developments in quantum computing?",
        "Explain the concept of attention mechanism in transformers"
    ]
    
    for query in queries:
        print(f"\n🤔 Query: {query}")
        print("-" * 60)
        
        result = atlas.query(query)
        
        print(f"\n💡 Answer:\n{result['answer']}")
        print(f"\n📊 Quality Score: {result['score']:.1f}/100")
        print(f"🔄 Iterations: {result['iterations']}")
        print("=" * 60)

if __name__ == "__main__":
    main()
