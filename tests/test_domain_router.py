from app.core.domain_router import DomainRouter


def test_routes_cuda_query():
    r = DomainRouter()
    result = r.route("write a CUDA kernel for matrix multiplication")
    assert result.domain == "cuda_kernels"
    assert result.confidence > 0.3


def test_routes_legal_query():
    r = DomainRouter()
    result = r.route("what is section 302 IPC BNS murder case")
    assert result.domain == "indian_legal"


def test_routes_business_query():
    r = DomainRouter()
    result = r.route("help me create a pitch deck for my startup")
    assert result.domain == "business_strategy"


def test_routes_space_query():
    r = DomainRouter()
    result = r.route("explain ISRO Chandrayaan mission details")
    assert result.domain == "space_astronomy"


def test_domain_prompt_not_empty_for_known_domains():
    r = DomainRouter()
    result = r.route("GST filing income tax audit India CA")
    assert result.domain_prompt != ""


def test_unknown_query_returns_general():
    r = DomainRouter()
    result = r.route("xyzzy undefined gibberish query")
    assert result.domain == "general_reasoning"
