import pandas as pd
from env.agents import Agent
from utils.dataprocessor import YAMLParser, GecadDataProcessor, DataProcessor

class Community:
    def __init__(self, 
                 agents_config,
                 scenarios_config,
                 problem_config,
                 data_file):

        self.parser = YAMLParser()
        self.agents_conf = self.parser.load_yaml(agents_config)
        self.scenarios_conf = self.parser.load_yaml(scenarios_config)
        self.problem_conf = self.parser.load_yaml(problem_config)

        # Make agent DataProcessor
        self.agent_processor = GecadDataProcessor(problem_config,
                                                  agents_config,
                                                  data_file)
        # Make community DataProcessor
        self.com_processor = DataProcessor()

        # Assemble agents
        self.assemble_agents()
        self.num_agents = len(self.agents)

        self.print_info()

        # Make community data
        self.com_data = self.com_processor.make_com_data(
            self.agents, 
            self.problem_conf['t_init'], 
            self.problem_conf['t_end'], 
            self.problem_conf['pv_factor'])

        # Get profiles and preferences from agents
        self.get_agents_prefs()

    def get_agent_obj(self, agent_id):
        return self.agents[agent_id]

    def print_info(self):
        print('Created an Energy Community with the following characteristics')
        print('number of agents: ', self.num_agents)

    def assemble_agents(self):
        """Assembles the agents and returns a dictionary of agent objects built from config files."""
        agents = {}
        for agent_id, agent_conf in self.agents_conf.items():
            agent_cls = self._get_class(agent_conf['agent_cls'])
            agents[agent_id] = agent_cls(agent_id, 
                                         self.agent_processor, 
                                         [],  # Empty list for appliances
                                         agent_conf)
        self.agents = agents

    def _get_class(self, class_name):
        """Retrieve a class object by its name."""
        return globals().get(class_name)

    def get_agents_prefs(self):
        agent_preferences = {}
        for agent in self.agents.values():
            agent_preferences[agent.id] = agent.get_params()
        prefs_df = pd.DataFrame.from_dict(agent_preferences, orient='index')
        self.com_prefs = prefs_df

    def get_tariffs(self, timeslot):
        df = pd.DataFrame()
        for agent in self.agents.values():
            df.loc[agent.id, 'tar_buy'] = agent.get_tariff(timeslot)
        return df

    def get_tariffs_by_mins(self, timeslot):
        df = pd.DataFrame()
        for agent in self.agents.values():
            df.loc[agent.id, 'tar_buy'] = agent.get_tar_by_mins(timeslot)
        return df

    def get_tariffs_indexed(self, indexes):
        df = pd.DataFrame()
        for aid in self.agents.keys():
            tar = self.agents[aid].tariff
            df.loc[aid] = tar
        return df
