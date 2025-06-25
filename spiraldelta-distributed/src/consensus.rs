/*!
Consensus and Coordination

Implements Raft consensus protocol for distributed coordination and leader election.
*/

use crate::*;
use anyhow::{Result, anyhow};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{RwLock, mpsc};
use tracing::{info, warn, error, debug, instrument};
use serde::{Serialize, Deserialize};
use std::time::{Duration, Instant};

/// Consensus service implementing Raft protocol
pub struct ConsensusService {
    /// Node information
    node_info: NodeInfo,
    
    /// Consensus configuration
    config: ConsensusConfig,
    
    /// Current Raft state
    raft_state: Arc<RwLock<RaftState>>,
    
    /// Peer nodes in the cluster
    peers: Arc<RwLock<HashMap<NodeId, PeerConnection>>>,
    
    /// Log entries for consensus
    log: Arc<RwLock<RaftLog>>,
    
    /// State machine for applying committed entries
    state_machine: Arc<RwLock<ClusterStateMachine>>,
    
    /// Message channels for Raft communication
    message_tx: mpsc::UnboundedSender<RaftMessage>,
    message_rx: Arc<RwLock<Option<mpsc::UnboundedReceiver<RaftMessage>>>>,
    
    /// Election timer
    election_timer: Arc<RwLock<Option<tokio::time::Interval>>>,
    
    /// Heartbeat timer for leaders
    heartbeat_timer: Arc<RwLock<Option<tokio::time::Interval>>>,
}

#[derive(Debug, Clone)]
struct RaftState {
    /// Current role (follower, candidate, leader)
    role: RaftRole,
    
    /// Current term
    current_term: u64,
    
    /// Node voted for in current term
    voted_for: Option<NodeId>,
    
    /// Current leader
    current_leader: Option<NodeId>,
    
    /// Index of highest log entry known to be committed
    commit_index: u64,
    
    /// Index of highest log entry applied to state machine
    last_applied: u64,
    
    /// For leaders: next index to send to each server
    next_index: HashMap<NodeId, u64>,
    
    /// For leaders: highest index known to be replicated on each server
    match_index: HashMap<NodeId, u64>,
}

#[derive(Debug, Clone, PartialEq)]
enum RaftRole {
    Follower,
    Candidate,
    Leader,
}

#[derive(Debug, Clone)]
struct PeerConnection {
    node_id: NodeId,
    address: std::net::SocketAddr,
    last_heartbeat: Option<Instant>,
    next_index: u64,
    match_index: u64,
}

/// Raft log for storing consensus entries
#[derive(Debug)]
struct RaftLog {
    /// Log entries
    entries: Vec<LogEntry>,
    
    /// Persistent log storage
    storage: Option<LogStorage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LogEntry {
    /// Log entry index
    index: u64,
    
    /// Term when entry was received by leader
    term: u64,
    
    /// Command for state machine
    command: ClusterCommand,
    
    /// Timestamp when entry was created
    timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum ClusterCommand {
    /// Add a node to the cluster
    AddNode(NodeInfo),
    
    /// Remove a node from the cluster
    RemoveNode(NodeId),
    
    /// Update cluster configuration
    UpdateConfig(ClusterConfig),
    
    /// Update shard assignment
    UpdateShards(Vec<ShardInfo>),
    
    /// No-op command for heartbeats
    NoOp,
}

/// State machine for applying committed log entries
#[derive(Debug)]
struct ClusterStateMachine {
    /// Current cluster topology
    topology: ClusterTopology,
    
    /// Applied log index
    last_applied_index: u64,
}

/// Storage for persistent Raft log
struct LogStorage {
    // In a real implementation, this would use RocksDB or similar
    // For now, we'll keep it simple
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum RaftMessage {
    /// Request vote for election
    RequestVote {
        term: u64,
        candidate_id: NodeId,
        last_log_index: u64,
        last_log_term: u64,
    },
    
    /// Response to vote request
    RequestVoteResponse {
        term: u64,
        vote_granted: bool,
    },
    
    /// Append entries (including heartbeat)
    AppendEntries {
        term: u64,
        leader_id: NodeId,
        prev_log_index: u64,
        prev_log_term: u64,
        entries: Vec<LogEntry>,
        leader_commit: u64,
    },
    
    /// Response to append entries
    AppendEntriesResponse {
        term: u64,
        success: bool,
        match_index: u64,
    },
    
    /// Install snapshot
    InstallSnapshot {
        term: u64,
        leader_id: NodeId,
        last_included_index: u64,
        last_included_term: u64,
        data: Vec<u8>,
    },
    
    /// Response to install snapshot
    InstallSnapshotResponse {
        term: u64,
    },
}

impl ConsensusService {
    /// Create a new consensus service
    pub async fn new(
        node_info: NodeInfo,
        config: ConsensusConfig,
        initial_peers: Vec<NodeInfo>,
    ) -> Result<Self> {
        let (message_tx, message_rx) = mpsc::unbounded_channel();
        
        let raft_state = Arc::new(RwLock::new(RaftState {
            role: RaftRole::Follower,
            current_term: 0,
            voted_for: None,
            current_leader: None,
            commit_index: 0,
            last_applied: 0,
            next_index: HashMap::new(),
            match_index: HashMap::new(),
        }));
        
        let peers = Arc::new(RwLock::new(
            initial_peers.into_iter()
                .filter(|peer| peer.node_id != node_info.node_id)
                .map(|peer| (peer.node_id.clone(), PeerConnection {
                    node_id: peer.node_id,
                    address: peer.address,
                    last_heartbeat: None,
                    next_index: 1,
                    match_index: 0,
                }))
                .collect()
        ));
        
        let log = Arc::new(RwLock::new(RaftLog {
            entries: vec![],
            storage: None,
        }));
        
        let state_machine = Arc::new(RwLock::new(ClusterStateMachine {
            topology: ClusterTopology::new(&ClusterConfig::default())?,
            last_applied_index: 0,
        }));
        
        Ok(Self {
            node_info,
            config,
            raft_state,
            peers,
            log,
            state_machine,
            message_tx,
            message_rx: Arc::new(RwLock::new(Some(message_rx))),
            election_timer: Arc::new(RwLock::new(None)),
            heartbeat_timer: Arc::new(RwLock::new(None)),
        })
    }
    
    /// Start the consensus service
    #[instrument(skip(self))]
    pub async fn start(&self) -> Result<()> {
        info!("Starting consensus service for node: {}", self.node_info.node_id);
        
        // Start election timer
        self.reset_election_timer().await;
        
        // Start message processing loop
        self.start_message_processing().await?;
        
        // Start consensus loop
        self.start_consensus_loop().await?;
        
        info!("Consensus service started");
        Ok(())
    }
    
    /// Propose a new cluster command
    #[instrument(skip(self))]
    pub async fn propose_command(&self, command: ClusterCommand) -> Result<u64> {
        let state = self.raft_state.read().await;
        
        if state.role != RaftRole::Leader {
            return Err(anyhow!("Only leaders can propose commands"));
        }
        
        let current_term = state.current_term;
        drop(state);
        
        // Create log entry
        let mut log = self.log.write().await;
        let next_index = log.entries.len() as u64 + 1;
        
        let entry = LogEntry {
            index: next_index,
            term: current_term,
            command,
            timestamp: chrono::Utc::now(),
        };
        
        log.entries.push(entry);
        
        info!("Proposed command at index {}", next_index);
        
        // Trigger replication to followers
        self.replicate_to_followers().await?;
        
        Ok(next_index)
    }
    
    /// Get current cluster leader
    pub async fn get_leader(&self) -> Option<NodeId> {
        let state = self.raft_state.read().await;
        state.current_leader.clone()
    }
    
    /// Check if this node is the leader
    pub async fn is_leader(&self) -> bool {
        let state = self.raft_state.read().await;
        state.role == RaftRole::Leader
    }
    
    /// Get current term
    pub async fn get_current_term(&self) -> u64 {
        let state = self.raft_state.read().await;
        state.current_term
    }
    
    async fn start_message_processing(&self) -> Result<()> {
        let mut message_rx = self.message_rx.write().await;
        let receiver = message_rx.take()
            .ok_or_else(|| anyhow!("Message receiver already taken"))?;
        drop(message_rx);
        
        let consensus = self.clone();
        tokio::spawn(async move {
            consensus.message_processing_loop(receiver).await;
        });
        
        Ok(())
    }
    
    async fn message_processing_loop(&self, mut receiver: mpsc::UnboundedReceiver<RaftMessage>) {
        while let Some(message) = receiver.recv().await {
            if let Err(e) = self.handle_raft_message(message).await {
                error!("Failed to handle Raft message: {}", e);
            }
        }
    }
    
    async fn handle_raft_message(&self, message: RaftMessage) -> Result<()> {
        match message {
            RaftMessage::RequestVote { term, candidate_id, last_log_index, last_log_term } => {
                self.handle_request_vote(term, candidate_id, last_log_index, last_log_term).await
            }
            RaftMessage::RequestVoteResponse { term, vote_granted } => {
                self.handle_request_vote_response(term, vote_granted).await
            }
            RaftMessage::AppendEntries { term, leader_id, prev_log_index, prev_log_term, entries, leader_commit } => {
                self.handle_append_entries(term, leader_id, prev_log_index, prev_log_term, entries, leader_commit).await
            }
            RaftMessage::AppendEntriesResponse { term, success, match_index } => {
                self.handle_append_entries_response(term, success, match_index).await
            }
            RaftMessage::InstallSnapshot { term, leader_id, last_included_index, last_included_term, data } => {
                self.handle_install_snapshot(term, leader_id, last_included_index, last_included_term, data).await
            }
            RaftMessage::InstallSnapshotResponse { term } => {
                self.handle_install_snapshot_response(term).await
            }
        }
    }
    
    async fn handle_request_vote(
        &self,
        term: u64,
        candidate_id: NodeId,
        last_log_index: u64,
        last_log_term: u64,
    ) -> Result<()> {
        let mut state = self.raft_state.write().await;
        
        // If term is greater, become follower and update term
        if term > state.current_term {
            state.current_term = term;
            state.voted_for = None;
            state.role = RaftRole::Follower;
            state.current_leader = None;
        }
        
        let vote_granted = term == state.current_term &&
            (state.voted_for.is_none() || state.voted_for.as_ref() == Some(&candidate_id)) &&
            self.is_candidate_log_up_to_date(last_log_index, last_log_term).await?;
        
        if vote_granted {
            state.voted_for = Some(candidate_id.clone());
            self.reset_election_timer().await;
        }
        
        // Send response
        self.send_message_to_peer(&candidate_id, RaftMessage::RequestVoteResponse {
            term: state.current_term,
            vote_granted,
        }).await?;
        
        Ok(())
    }
    
    async fn handle_request_vote_response(&self, term: u64, vote_granted: bool) -> Result<()> {
        let mut state = self.raft_state.write().await;
        
        if term > state.current_term {
            state.current_term = term;
            state.voted_for = None;
            state.role = RaftRole::Follower;
            state.current_leader = None;
            return Ok(());
        }
        
        if state.role == RaftRole::Candidate && term == state.current_term && vote_granted {
            // Count votes
            let peers = self.peers.read().await;
            let total_nodes = peers.len() + 1; // +1 for self
            let majority = total_nodes / 2 + 1;
            
            // For simplicity, we'll assume we got enough votes
            // In a real implementation, you'd track votes properly
            if true { // Placeholder for vote counting
                // Become leader
                state.role = RaftRole::Leader;
                state.current_leader = Some(self.node_info.node_id.clone());
                
                // Initialize leader state
                let last_log_index = {
                    let log = self.log.read().await;
                    log.entries.len() as u64
                };
                
                for peer_id in peers.keys() {
                    state.next_index.insert(peer_id.clone(), last_log_index + 1);
                    state.match_index.insert(peer_id.clone(), 0);
                }
                
                info!("Node {} became leader for term {}", self.node_info.node_id, state.current_term);
                
                // Start sending heartbeats
                self.start_heartbeat_timer().await;
                
                // Send initial heartbeat
                drop(state);
                self.send_heartbeats().await?;
            }
        }
        
        Ok(())
    }
    
    async fn handle_append_entries(
        &self,
        term: u64,
        leader_id: NodeId,
        prev_log_index: u64,
        prev_log_term: u64,
        entries: Vec<LogEntry>,
        leader_commit: u64,
    ) -> Result<()> {
        let mut state = self.raft_state.write().await;
        
        // Reset election timer since we received from leader
        self.reset_election_timer().await;
        
        // If term is greater, become follower
        if term > state.current_term {
            state.current_term = term;
            state.voted_for = None;
            state.role = RaftRole::Follower;
        }
        
        state.current_leader = Some(leader_id.clone());
        
        let success = if term < state.current_term {
            false
        } else {
            // Check if log contains entry at prev_log_index with matching term
            let log_ok = self.check_log_consistency(prev_log_index, prev_log_term).await?;
            
            if log_ok {
                // Append new entries
                if !entries.is_empty() {
                    let mut log = self.log.write().await;
                    
                    // Remove conflicting entries
                    if prev_log_index < log.entries.len() as u64 {
                        log.entries.truncate(prev_log_index as usize);
                    }
                    
                    // Append new entries
                    log.entries.extend(entries);
                }
                
                // Update commit index
                if leader_commit > state.commit_index {
                    let log = self.log.read().await;
                    state.commit_index = std::cmp::min(leader_commit, log.entries.len() as u64);
                }
                
                true
            } else {
                false
            }
        };
        
        let match_index = if success {
            let log = self.log.read().await;
            log.entries.len() as u64
        } else {
            0
        };
        
        // Send response
        self.send_message_to_peer(&leader_id, RaftMessage::AppendEntriesResponse {
            term: state.current_term,
            success,
            match_index,
        }).await?;
        
        Ok(())
    }
    
    async fn handle_append_entries_response(&self, term: u64, success: bool, match_index: u64) -> Result<()> {
        let mut state = self.raft_state.write().await;
        
        if term > state.current_term {
            state.current_term = term;
            state.voted_for = None;
            state.role = RaftRole::Follower;
            state.current_leader = None;
            return Ok(());
        }
        
        if state.role == RaftRole::Leader && term == state.current_term {
            if success {
                // Update match and next indices
                // In a real implementation, you'd track which peer sent this response
                // For now, we'll update all peers (simplified)
                for peer_id in state.match_index.keys().cloned().collect::<Vec<_>>() {
                    if let Some(current_match) = state.match_index.get(&peer_id) {
                        if match_index > *current_match {
                            state.match_index.insert(peer_id.clone(), match_index);
                            state.next_index.insert(peer_id, match_index + 1);
                        }
                    }
                }
                
                // Update commit index if majority of servers have replicated
                self.update_commit_index().await?;
            }
        }
        
        Ok(())
    }
    
    async fn handle_install_snapshot(
        &self,
        _term: u64,
        _leader_id: NodeId,
        _last_included_index: u64,
        _last_included_term: u64,
        _data: Vec<u8>,
    ) -> Result<()> {
        // Placeholder for snapshot installation
        Ok(())
    }
    
    async fn handle_install_snapshot_response(&self, _term: u64) -> Result<()> {
        // Placeholder for snapshot response handling
        Ok(())
    }
    
    async fn start_consensus_loop(&self) -> Result<()> {
        let consensus = self.clone();
        tokio::spawn(async move {
            consensus.consensus_loop().await;
        });
        Ok(())
    }
    
    async fn consensus_loop(&self) {
        let mut election_interval = tokio::time::interval(
            Duration::from_millis(self.config.election_timeout_ms)
        );
        
        loop {
            election_interval.tick().await;
            
            let state = self.raft_state.read().await;
            match state.role {
                RaftRole::Follower => {
                    // Check if election timeout has elapsed
                    // For simplicity, we'll periodically check if we should start election
                    drop(state);
                    if self.should_start_election().await {
                        if let Err(e) = self.start_election().await {
                            error!("Failed to start election: {}", e);
                        }
                    }
                }
                RaftRole::Candidate => {
                    // Election timeout, start new election
                    drop(state);
                    if let Err(e) = self.start_election().await {
                        error!("Failed to start election: {}", e);
                    }
                }
                RaftRole::Leader => {
                    // Apply committed entries to state machine
                    drop(state);
                    if let Err(e) = self.apply_committed_entries().await {
                        error!("Failed to apply committed entries: {}", e);
                    }
                }
            }
        }
    }
    
    async fn should_start_election(&self) -> bool {
        // Simplified logic - in real implementation, this would check election timer
        rand::random::<f64>() < 0.01 // 1% chance per tick
    }
    
    async fn start_election(&self) -> Result<()> {
        let mut state = self.raft_state.write().await;
        
        // Increment term and vote for self
        state.current_term += 1;
        state.voted_for = Some(self.node_info.node_id.clone());
        state.role = RaftRole::Candidate;
        state.current_leader = None;
        
        let term = state.current_term;
        drop(state);
        
        info!("Starting election for term {}", term);
        
        // Reset election timer
        self.reset_election_timer().await;
        
        // Get last log info
        let (last_log_index, last_log_term) = {
            let log = self.log.read().await;
            if let Some(last_entry) = log.entries.last() {
                (last_entry.index, last_entry.term)
            } else {
                (0, 0)
            }
        };
        
        // Send RequestVote to all peers
        let peers = self.peers.read().await;
        for peer_id in peers.keys() {
            self.send_message_to_peer(peer_id, RaftMessage::RequestVote {
                term,
                candidate_id: self.node_info.node_id.clone(),
                last_log_index,
                last_log_term,
            }).await?;
        }
        
        Ok(())
    }
    
    async fn send_heartbeats(&self) -> Result<()> {
        let state = self.raft_state.read().await;
        if state.role != RaftRole::Leader {
            return Ok(());
        }
        
        let term = state.current_term;
        drop(state);
        
        let peers = self.peers.read().await;
        for peer_id in peers.keys() {
            // Send empty append entries as heartbeat
            self.send_message_to_peer(peer_id, RaftMessage::AppendEntries {
                term,
                leader_id: self.node_info.node_id.clone(),
                prev_log_index: 0,
                prev_log_term: 0,
                entries: vec![],
                leader_commit: 0,
            }).await?;
        }
        
        Ok(())
    }
    
    async fn replicate_to_followers(&self) -> Result<()> {
        // Simplified replication - in real implementation, this would be more sophisticated
        self.send_heartbeats().await
    }
    
    async fn send_message_to_peer(&self, peer_id: &NodeId, message: RaftMessage) -> Result<()> {
        // In a real implementation, this would send the message over network
        // For now, we'll just log it
        debug!("Sending message to {}: {:?}", peer_id, message);
        Ok(())
    }
    
    async fn is_candidate_log_up_to_date(&self, last_log_index: u64, last_log_term: u64) -> Result<bool> {
        let log = self.log.read().await;
        
        if let Some(last_entry) = log.entries.last() {
            Ok(last_log_term > last_entry.term || 
               (last_log_term == last_entry.term && last_log_index >= last_entry.index))
        } else {
            Ok(true) // Empty log is always up to date
        }
    }
    
    async fn check_log_consistency(&self, prev_log_index: u64, prev_log_term: u64) -> Result<bool> {
        let log = self.log.read().await;
        
        if prev_log_index == 0 {
            return Ok(true); // Base case
        }
        
        if prev_log_index > log.entries.len() as u64 {
            return Ok(false); // Log too short
        }
        
        if let Some(entry) = log.entries.get((prev_log_index - 1) as usize) {
            Ok(entry.term == prev_log_term)
        } else {
            Ok(false)
        }
    }
    
    async fn update_commit_index(&self) -> Result<()> {
        // Simplified commit index update
        // In real implementation, this would check majority replication
        Ok(())
    }
    
    async fn apply_committed_entries(&self) -> Result<()> {
        let state = self.raft_state.read().await;
        let commit_index = state.commit_index;
        let last_applied = state.last_applied;
        drop(state);
        
        if commit_index > last_applied {
            let log = self.log.read().await;
            let mut state_machine = self.state_machine.write().await;
            
            for i in (last_applied + 1)..=commit_index {
                if let Some(entry) = log.entries.get((i - 1) as usize) {
                    self.apply_command_to_state_machine(&mut state_machine, &entry.command).await?;
                    state_machine.last_applied_index = i;
                }
            }
            
            drop(state_machine);
            drop(log);
            
            let mut state = self.raft_state.write().await;
            state.last_applied = commit_index;
        }
        
        Ok(())
    }
    
    async fn apply_command_to_state_machine(
        &self,
        state_machine: &mut ClusterStateMachine,
        command: &ClusterCommand,
    ) -> Result<()> {
        match command {
            ClusterCommand::AddNode(node_info) => {
                state_machine.topology.add_node(node_info.clone())?;
                info!("Applied command: added node {}", node_info.node_id);
            }
            ClusterCommand::RemoveNode(node_id) => {
                state_machine.topology.remove_node(node_id)?;
                info!("Applied command: removed node {}", node_id);
            }
            ClusterCommand::UpdateConfig(_config) => {
                info!("Applied command: updated cluster config");
            }
            ClusterCommand::UpdateShards(_shards) => {
                info!("Applied command: updated shards");
            }
            ClusterCommand::NoOp => {
                // No-op, used for heartbeats
            }
        }
        Ok(())
    }
    
    async fn reset_election_timer(&self) {
        // In a real implementation, this would reset the actual election timer
        // For now, we'll just log it
        debug!("Reset election timer");
    }
    
    async fn start_heartbeat_timer(&self) {
        let consensus = self.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(
                Duration::from_millis(consensus.config.heartbeat_interval_ms)
            );
            
            loop {
                interval.tick().await;
                
                let state = consensus.raft_state.read().await;
                if state.role == RaftRole::Leader {
                    drop(state);
                    if let Err(e) = consensus.send_heartbeats().await {
                        error!("Failed to send heartbeats: {}", e);
                    }
                } else {
                    break; // Stop heartbeats if no longer leader
                }
            }
        });
    }
}

impl Clone for ConsensusService {
    fn clone(&self) -> Self {
        Self {
            node_info: self.node_info.clone(),
            config: self.config.clone(),
            raft_state: self.raft_state.clone(),
            peers: self.peers.clone(),
            log: self.log.clone(),
            state_machine: self.state_machine.clone(),
            message_tx: self.message_tx.clone(),
            message_rx: self.message_rx.clone(),
            election_timer: self.election_timer.clone(),
            heartbeat_timer: self.heartbeat_timer.clone(),
        }
    }
}

use rand;