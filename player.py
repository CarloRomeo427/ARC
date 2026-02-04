"""
Player classes for persistent and transient (raid) players.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Player:
    """Persistent player with running statistics."""
    id: int
    aggression: float  # Base aggression (initialized, evolves)
    
    # Running stats (start at 0, filled during experiment)
    total_raids: int = 0
    total_extractions: int = 0
    total_deaths: int = 0
    total_kills: int = 0
    total_stash: float = 0.0
    total_damage_dealt: float = 0.0
    total_damage_received: float = 0.0
    
    # Running aggression tracking
    aggression_sum: float = 0.0
    aggression_count: int = 0
    
    @property
    def running_aggression(self) -> float:
        if self.aggression_count == 0:
            return self.aggression
        return self.aggression_sum / self.aggression_count
    
    @property
    def classification(self) -> str:
        ra = self.running_aggression
        if ra < 0.4:
            return "passive"
        elif ra > 0.6:
            return "aggressive"
        return "neutral"
    
    @property
    def extraction_rate(self) -> float:
        if self.total_raids == 0:
            return 0.0
        return self.total_extractions / self.total_raids
    
    def get_raid_aggression(self, noise_std: float = 0.05) -> float:
        """Get aggression for raid with noise."""
        noisy = self.aggression + np.random.normal(0, noise_std)
        return np.clip(noisy, 0.0, 1.0)
    
    def record_raid(self, extracted: bool, stash: float, damage_dealt: float,
                    damage_received: float, kills: int, aggression_used: float):
        """Update running stats after raid."""
        self.total_raids += 1
        self.total_damage_dealt += damage_dealt
        self.total_damage_received += damage_received
        self.total_kills += kills
        self.aggression_sum += aggression_used
        self.aggression_count += 1
        
        if extracted:
            self.total_extractions += 1
            self.total_stash += stash
        else:
            self.total_deaths += 1
    
    def update_aggression(self, extracted: bool, kills: int, damage_dealt: float,
                          damage_received: float, aggression_used: float,
                          learning_rate: float = 0.03):
        """Evolve base aggression."""
        delta = 0.0
        
        if extracted:
            if kills > 0:
                delta += learning_rate * (0.4 + kills * 0.15)
            else:
                delta -= learning_rate * 0.6
        else:
            if aggression_used > 0.5:
                delta -= learning_rate * 0.5
            else:
                delta -= learning_rate * 0.3
        
        if damage_dealt > 100:
            delta += learning_rate * 0.2
        elif damage_dealt < 30:
            delta -= learning_rate * 0.2
        
        if 0.4 < self.aggression < 0.6:
            delta += learning_rate * 0.1 * (0.5 - self.aggression)
        
        self.aggression = np.clip(self.aggression + delta, 0.0, 1.0)
    
    def copy(self) -> 'Player':
        """Deep copy."""
        p = Player(id=self.id, aggression=self.aggression)
        p.total_raids = self.total_raids
        p.total_extractions = self.total_extractions
        p.total_deaths = self.total_deaths
        p.total_kills = self.total_kills
        p.total_stash = self.total_stash
        p.total_damage_dealt = self.total_damage_dealt
        p.total_damage_received = self.total_damage_received
        p.aggression_sum = self.aggression_sum
        p.aggression_count = self.aggression_count
        return p


class PlayerPool:
    """Pool of persistent players."""
    
    def __init__(self, num_players: int, seed: int):
        """Initialize with diverse aggression scores."""
        np.random.seed(seed)
        self.players: Dict[int, Player] = {}
        
        for i in range(num_players):
            aggr = np.random.uniform(0.05, 0.95)
            self.players[i] = Player(id=i, aggression=aggr)
        
        np.random.seed(None)
    
    def get_all_players(self) -> List[Player]:
        return list(self.players.values())
    
    def get_players_by_classification(self) -> Dict[str, List[Player]]:
        groups = {'passive': [], 'neutral': [], 'aggressive': []}
        for p in self.players.values():
            groups[p.classification].append(p)
        return groups
    
    def get_stats(self) -> dict:
        players = self.get_all_players()
        aggr = [p.running_aggression for p in players]
        groups = self.get_players_by_classification()
        
        return {
            'aggression_mean': np.mean(aggr),
            'aggression_std': np.std(aggr),
            'passive_count': len(groups['passive']),
            'neutral_count': len(groups['neutral']),
            'aggressive_count': len(groups['aggressive']),
            'total_extractions': sum(p.total_extractions for p in players),
            'total_deaths': sum(p.total_deaths for p in players),
            'avg_extraction_rate': np.mean([p.extraction_rate for p in players]) if players[0].total_raids > 0 else 0,
        }
    
    def copy(self) -> 'PlayerPool':
        """Deep copy the pool."""
        new_pool = PlayerPool.__new__(PlayerPool)
        new_pool.players = {pid: p.copy() for pid, p in self.players.items()}
        return new_pool


@dataclass
class RaidPlayer:
    """Transient player for single raid."""
    id: int
    persistent_id: int
    x: float
    y: float
    aggression: float
    
    hp: float = 100.0
    stash: float = 0.0
    alive: bool = True
    extracted: bool = False
    speed: float = 2.0
    
    target_x: float = 0.0
    target_y: float = 0.0
    target_type: str = "loot"
    
    extraction_ticks: int = 0
    extracting_at: Optional[str] = None
    
    combat_cooldown: int = 0
    no_fight_cooldowns: dict = field(default_factory=dict)
    in_combat_with: Optional[int] = None
    is_attacker: bool = False
    combat_ticks: int = 0
    
    damage_dealt: float = 0.0
    damage_received: float = 0.0
    kills: int = 0
    
    def is_alive(self) -> bool:
        return self.alive and self.hp > 0
    
    def is_extracting(self) -> bool:
        return self.extracting_at is not None
    
    def is_in_combat(self) -> bool:
        return self.in_combat_with is not None
    
    def can_fight(self, other_id: int) -> bool:
        if self.combat_cooldown > 0 or self.in_combat_with is not None:
            return False
        return other_id not in self.no_fight_cooldowns or self.no_fight_cooldowns[other_id] <= 0
    
    def decide_to_fight(self) -> bool:
        return np.random.random() < self.aggression
    
    def deal_damage(self) -> float:
        return np.random.uniform(1 + 4 * self.aggression, 5 + 15 * self.aggression)
    
    def take_damage(self, dmg: float):
        self.hp -= dmg
        self.damage_received += dmg
        if self.hp <= 0:
            self.hp = 0
            self.alive = False
    
    def heal(self, amount: float):
        self.hp = min(100.0, self.hp + amount)
    
    def distance_to(self, x: float, y: float) -> float:
        return np.sqrt((self.x - x)**2 + (self.y - y)**2)
    
    def move_toward(self, tx: float, ty: float, map_radius: float):
        dx, dy = tx - self.x, ty - self.y
        dist = np.sqrt(dx**2 + dy**2)
        if dist < self.speed:
            self.x, self.y = tx, ty
        else:
            self.x += self.speed * dx / dist
            self.y += self.speed * dy / dist
        dist_center = np.sqrt(self.x**2 + self.y**2)
        if dist_center > map_radius:
            scale = (map_radius - 0.1) / dist_center
            self.x *= scale
            self.y *= scale
    
    def update_cooldowns(self):
        if self.combat_cooldown > 0:
            self.combat_cooldown -= 1
        expired = [pid for pid, cd in self.no_fight_cooldowns.items() if cd <= 1]
        for pid in expired:
            del self.no_fight_cooldowns[pid]
        for pid in self.no_fight_cooldowns:
            self.no_fight_cooldowns[pid] -= 1
    
    def start_combat(self, other_id: int, as_attacker: bool):
        self.in_combat_with = other_id
        self.is_attacker = as_attacker
        self.combat_ticks = 0
    
    def end_combat(self):
        self.in_combat_with = None
        self.is_attacker = False
        self.combat_ticks = 0
    
    def set_no_fight_cooldown(self, other_id: int, steps: int = 30):
        self.no_fight_cooldowns[other_id] = steps
    
    def to_result(self) -> dict:
        """Convert to result dict."""
        return {
            'persistent_id': self.persistent_id,
            'extracted': self.extracted,
            'stash': self.stash if self.extracted else 0,
            'damage_dealt': self.damage_dealt,
            'damage_received': self.damage_received,
            'kills': self.kills,
            'aggression_used': self.aggression
        }
