import typing

import pacai.agents.greedy
import pacai.core.action
import pacai.core.agent
import pacai.core.board
import pacai.core.gamestate
import pacai.core.features
import pacai.search.distance
import pacai.pacman.board
import pacai.capture.gamestate

GHOST_IGNORE_RANGE: float = 2.5

def create_team() -> list[pacai.core.agentinfo.AgentInfo]:
    """
    Get the agent information that will be used to create a capture team.
    """

    agent1_info = pacai.core.agentinfo.AgentInfo(name = f"{__name__}.DefensiveAgent")
    agent2_info = pacai.core.agentinfo.AgentInfo(name = f"{__name__}.OffensiveAgent")

    return [agent1_info, agent2_info]

class DefensiveAgent(pacai.agents.greedy.GreedyFeatureAgent):
    """
    A capture agent that prioritizes defending its own territory.
    """

    def __init__(self,
            override_weights: dict[str, float] | None = None,
            **kwargs: typing.Any) -> None:
        kwargs['feature_extractor_func'] = _extract_baseline_defensive_features
        super().__init__(**kwargs)

        self._distances: pacai.search.distance.DistancePreComputer = pacai.search.distance.DistancePreComputer()
        """ Precompute distances. """

        # Set base weights.
        self.weights['on_home_side'] = 100.0
        self.weights['stopped'] = -100.0
        self.weights['reverse'] = -2.0
        self.weights['num_invaders'] = -1000.0
        self.weights['distance_to_invader'] = -40.0
        self.weights['distance_to_ghost_squared'] = 0.0
        self.weights['food_count'] = -10.0
        self.weights['capsules'] = 0
        self.weights['food_distance'] = -5.0
        self.weights['keep_at_border_food_distance'] = -0.5
        self.weights['distance_to_opp'] = -2.0
        self.weights['distance_to_ally'] = 2.0
        self.weights['offense_food_distance'] = -1.5   # move toward enemy food
        self.weights['attack_mode'] = 20.0             # bonus for being in attack mode

        if (override_weights is None):
            override_weights = {}

        for (key, weight) in override_weights.items():
            self.weights[key] = weight

    def game_start(self, initial_state: pacai.core.gamestate.GameState) -> None:
        self._distances.compute(initial_state.board)

class OffensiveAgent(pacai.agents.greedy.GreedyFeatureAgent):
    """
    A capture agent that prioritizes defending its own territory.
    """

    def __init__(self,
            override_weights: dict[str, float] | None = None,
            **kwargs: typing.Any) -> None:
        kwargs['feature_extractor_func'] = _extract_baseline_offensive_features
        super().__init__(**kwargs)

        self._distances: pacai.search.distance.DistancePreComputer = pacai.search.distance.DistancePreComputer()
        """ Precompute distances. """

        # Set base weights.
        self.weights['score'] = 100.0
        self.weights['distance_to_food'] = -1.5
        self.weights['distance_to_ghost'] = 10.0
        self.weights['distance_to_ghost_squared'] = 0.0
        self.weights['capsules'] = 6.0
        self.weights['on_home_side'] = -30.0
        self.weights['distance_to_invader'] = 0.0
        self.weights["distance_to_danger"] = -15.0
        self.weights["distance_to_scared_ghost"] = 0.0
        self.weights['distance_to_ally'] = 1.0
        self.weights["repeat_position"] = -15.0
        self.weights["reverse"] = -10.0

        if (override_weights is None):
            override_weights = {}

        for (key, weight) in override_weights.items():
            self.weights[key] = weight

    def game_start(self, initial_state: pacai.core.gamestate.GameState) -> None:
        self._distances.compute(initial_state.board)

def _extract_baseline_defensive_features(
        state: pacai.core.gamestate.GameState,
        action: pacai.core.action.Action,
        agent: pacai.core.agent.Agent | None = None,
        **kwargs: typing.Any) -> pacai.core.features.FeatureDict:
    agent = typing.cast(DefensiveAgent, agent)
    state = typing.cast(pacai.capture.gamestate.GameState, state)

    features: pacai.core.features.FeatureDict = pacai.core.features.FeatureDict()

    current_position = state.get_agent_position(agent.agent_index)
    if (current_position is None):
        # We are dead and waiting to respawn.
        return features
    invader_positions = state.get_invader_positions(agent_index = agent.agent_index)
    num_invaders = len(invader_positions)

    opp_pos = state.get_opponent_positions(agent.agent_index)
    min_opp_distance = None
    if (len(opp_pos) > 0):
        opp_distances = [agent._distances.get_distance(current_position, opp)
                         for opp in opp_pos.values()]
        valid_opp_distances = [d for d in opp_distances if d is not None]
        if valid_opp_distances:
            min_opp_distance = min(valid_opp_distances)
            features['distance_to_opp'] = min_opp_distance

    SAFE_ATTACK_DISTANCE = 10.0  # tune this

    safe_to_attack = (num_invaders == 0) and (
        (min_opp_distance is None) or (min_opp_distance > SAFE_ATTACK_DISTANCE)
    )

    # when safe to attack, dontt reward staying home
    features['on_home_side'] = 0 if safe_to_attack else int(state.is_ghost(agent_index = agent.agent_index))

    # Prefer moving over stopping.
    features['stopped'] = int(action == pacai.core.action.STOP)

    # We don't like any invaders on our side.
    features['num_invaders'] = num_invaders

    # Note the side of the board we are on.
    features['on_home_side'] = 0 if safe_to_attack else int(state.is_ghost(agent_index = agent.agent_index))

    # Prefer moving over stopping.
    features['stopped'] = int(action == pacai.core.action.STOP)

    # getting where fellow teamates are to bias away from them and cover more ground
    ally_positions = state.get_ally_positions(agent_index=agent.agent_index)
    if ally_positions:
        ally_dists = [agent._distances.get_distance(current_position, pos) for pos in ally_positions.values()]
        features['distance_to_ally'] = min(d for d in ally_dists if d is not None)
    else:
        features['distance_to_ally'] = 0

    # COMMENTED out since it could be benificial to turning around and running away!!
    # agent_actions = state.get_agent_actions(agent.agent_index)
    # if (len(agent_actions) > 1):
    #     features['reverse'] = int(action == state.get_reverse_action(agent_actions[-2]))

    # We don't like any invaders on our side.
    
    if safe_to_attack:
        attack_food_positions = state.get_food(agent_index = agent.agent_index)
        if (len(attack_food_positions) > 0):
            attack_food_distances = [
                agent._distances.get_distance(current_position, food_pos)
                for food_pos in attack_food_positions
            ]
            valid_attack_distances = [d for d in attack_food_distances if d is not None]
            if valid_attack_distances:
                features['attack_food_distance'] = min(valid_attack_distances)
        features['attack_mode'] = 1.0
    else:
        features['attack_food_distance'] = 0.0
        features['attack_mode'] = 0.0
        
    # Here, I'm looking to protect last remaining food (<10) and have defense hover over it so invaders will be eaten before they can get to it
    team_mod = state._team_modifier(agent.agent_index)
    our_food = state.get_food(team_modifier = -team_mod)
    if len(invader_positions) == 0:
        if (len(our_food) > 0) and (len(our_food) < 10):
            food_distances = [agent._distances.get_distance(current_position, food_pos) for food_pos in our_food]
            features['food_distance'] = min(distance for distance in food_distances if (distance is not None))
        # if there is more food, instead of hovering/circling around the food, keep at the border to catch invaders early
        else:
            their_food = state.get_food(team_modifier = team_mod)
            if (len(their_food) > 0):
                their_food_distances = [agent._distances.get_distance(current_position, food_pos) for food_pos in their_food]
                features['keep_at_border_food_distance'] = min(distance for distance in their_food_distances if (distance is not None))

    # Hunt down the closest invader!
    if (len(invader_positions) > 0):
        invader_distances = [agent._distances.get_distance(current_position, invader_position) for invader_position in invader_positions.values()]
        features['distance_to_invader'] = min(distance for distance in invader_distances if (distance is not None))
        
    opp_pos = state.get_opponent_positions(agent.agent_index)
    if (len(opp_pos) > 0):
        opp_distances = [agent._distances.get_distance(current_position, opp) for opp in opp_pos.values()]
        features['distance_to_opp'] = min(distance for distance in opp_distances if (distance is not None))

    # keeping defense hovering around our capsule so opps dont eat it and make us vulnerable to being scared
    # TOOK THIS OUT -- would cause a lot of losses since it would hover in corners and opps would eat all food
    all_capsule_positions = state.board.get_marker_positions(pacai.pacman.board.MARKER_CAPSULE)
    # team_mod = state._team_modifier(agent.agent_index)
    our_caps = []
    for c in all_capsule_positions:
        if team_mod == state._team_side(agent.agent_index, c):
            our_caps.append(c)
    if (len(our_caps) > 0):
        distances_to_caps = []
        for capsule_pos in our_caps:
            distances_to_caps.append(agent._distances.get_distance(current_position, capsule_pos))
        features["capsules"] = min(capsule for capsule in distances_to_caps if (capsule is not None))
    
    # dont run at intruder if scared
    if state.is_scared(agent_index=agent.agent_index):
        features['distance_to_invader'] = 10.0
        features['distance_to_opp'] = 10.0
    
    return features

def _extract_baseline_offensive_features(
        state: pacai.core.gamestate.GameState,
        action: pacai.core.action.Action,
        agent: pacai.core.agent.Agent | None = None,
        **kwargs: typing.Any) -> pacai.core.features.FeatureDict:
    agent = typing.cast(OffensiveAgent, agent)
    state = typing.cast(pacai.capture.gamestate.GameState, state)
    # parameter to make sure if off is pac or no
    pacman = state.is_pacman(agent_index = agent.agent_index)
    features: pacai.core.features.FeatureDict = pacai.core.features.FeatureDict()
    features['score'] = state.get_normalized_score(agent.agent_index)

    # Note the side of the board we are on.
    features['on_home_side'] = int(state.is_ghost(agent_index = agent.agent_index))

    # Prefer moving over stopping.
    features['stopped'] = int(action == pacai.core.action.STOP)

    # Prefer not turning around.
    # Remember that the state we get is already a successor, so we have to look two actions back.
    agent_actions = state.get_agent_actions(agent.agent_index)
    if (len(agent_actions) > 1):
        features['reverse'] = int(action == state.get_reverse_action(agent_actions[-2]))
    
    current_position = state.get_agent_position(agent.agent_index)
    if (current_position is None):
        # We are dead and waiting to respawn.
        return features
    
    # again, try not to be near ally, ally's got it covered
    ally_positions = state.get_ally_positions(agent_index=agent.agent_index)
    if ally_positions:
        ally_dists = [agent._distances.get_distance(current_position, pos) for pos in ally_positions.values()]
        features['distance_to_ally'] = min(d for d in ally_dists if d is not None)
    else:
        features['distance_to_ally'] = 0

    # handling food
    food_positions = state.get_food(agent_index = agent.agent_index)
    if (len(food_positions) > 0):
        food_distances = [agent._distances.get_distance(current_position, food_position) for food_position in food_positions]
        features['distance_to_food'] = min(distance for distance in food_distances if (distance is not None))
    else:
        # There is no food left, give a large score.
        features['distance_to_food'] = -100000

    # mostly handling positions here and danger in proximity to opps
    if pacman:
        scared_ghost_pos = state.get_scared_opponent_positions(agent_index = agent.agent_index)
        ghost_positions = state.get_nonscared_opponent_positions(agent_index = agent.agent_index)
        # distance to ghosts
        if (len(ghost_positions) > 0):
            ghost_distances = [agent._distances.get_distance(current_position, ghost_position) for ghost_position in ghost_positions.values()]
            closest_ghost_distance = min(d for d in ghost_distances if d is not None)
            features['distance_to_ghost'] = min(distance for distance in ghost_distances if (distance is not None))
            if (features['distance_to_ghost'] > GHOST_IGNORE_RANGE):
                features['distance_to_ghost'] = 10000
            # if closer than ghost ignore range, food becomes less of a priority vs running away & sense of danger
            else:
                features['distance_to_food'] = -1
                # giving a sense of danger the closer you get(while keeping 0.5 as a base)... if close(er) to opps, run away!!
                features['distance_to_danger'] = 10.0 / max(0.5, closest_ghost_distance)

            features['distance_to_ghost_squared'] = features['distance_to_ghost'] ** 2
            
        if (len(scared_ghost_pos) > 0):
            scared_ghost_dists = [agent._distances.get_distance(current_position, scared_position) for scared_position in scared_ghost_pos.values()]
            # avoid scared ghosts (though not a lot) to avoid resetting timer
            features['distance_to_scared_ghost'] = min(distance for distance in scared_ghost_dists if (distance is not None))

    # get opponent capsules otherwise offense will hover on its own! Get opp then eat so enemy is ghost
    all_capsule_positions = state.board.get_marker_positions(pacai.pacman.board.MARKER_CAPSULE)
    team_mod = state._team_modifier(agent.agent_index)
    opp_caps = []
    for c in all_capsule_positions:
        if team_mod != state._team_side(agent.agent_index, c):
            opp_caps.append(c)
    if (len(opp_caps) > 0):
        distance_to_markers = [agent._distances.get_distance(current_position, capsule_position) for capsule_position in opp_caps]
        valid_capsule = [d for d in distance_to_markers if d is not None]
        if valid_capsule:
            # closer capusiles give larger feature*weight.
            features['capsules'] = -min(valid_capsule)
    else:
        # all finished big reward
        features['capsules'] = 0
    
    # scared_ghost_pos = state.get_scared_opponent_positions(agent_index = agent.agent_index)
    # if state.is_pa
            
    return features