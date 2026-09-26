def test_fake_agent_contract(fake_agent):
    fake_agent.act("x")
    assert fake_agent.acted == ["x"]
    assert fake_agent.model_info() == "fake-model-info"
