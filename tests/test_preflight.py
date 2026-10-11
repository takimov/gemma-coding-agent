from gemma_grpo.preflight import _valid_agent_name


def test_agent_name_matches_google_adk_identifier_rules():
    assert _valid_agent_name("gemma_grpo_agent")
    assert _valid_agent_name("_agent2")
    assert not _valid_agent_name("gemma-grpo-agent")
    assert not _valid_agent_name("2agent")
    assert not _valid_agent_name("agent name")
