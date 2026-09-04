<template>
  <div class="history-container glass-card full-width">
    <div class="section-header">
      <span class="icon">[List]</span>
      <h2>Recent Evaluations</h2>
    </div>
    
    <div v-if="history.length === 0" class="empty-state">
      <div class="icon">[Chart]</div>
      <p>Evaluation history will appear here.</p>
    </div>
    
    <table v-else class="history-table">
      <thead>
        <tr>
          <th>Time</th>
          <th>User ID</th>
          <th>Amount</th>
          <th>Score</th>
          <th>Routing</th>
          <th>Action</th>
        </tr>
      </thead>
      <transition-group name="fade" tag="tbody">
        <tr v-for="item in history" :key="item.id">
          <td>{{ formatTime(item.timestamp) }}</td>
          <td style="font-family: monospace;">{{ item.payload.user_id }}</td>
          <td>₹{{ item.payload.order_value.toLocaleString() }}</td>
          <td>
            <span class="score-pill" :class="getScoreClass(item.response.risk_score)">
              {{ item.response.risk_score.toFixed(4) }}
            </span>
          </td>
          <td>
            <span class="routing-badge" :class="item.response.routing">
              {{ item.response.routing === 'ml_pipeline' ? 'ML' : 'Rule' }}
            </span>
          </td>
          <td>
            <span style="font-size: 0.8rem; font-weight: 600;" :style="{ color: getActionColor(item.response.action) }">
              {{ formatAction(item.response.action) }}
            </span>
          </td>
        </tr>
      </transition-group>
    </table>
  </div>
</template>

<script>
export default {
  name: 'HistoryTable',
  props: {
    history: {
      type: Array,
      default: () => []
    }
  },
  methods: {
    formatTime(timestamp) {
      return new Date(timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
    },
    getScoreClass(score) {
      if (score < 0.3) return 'low'
      if (score < 0.6) return 'medium'
      if (score < 0.8) return 'high'
      return 'critical'
    },
    getActionColor(action) {
      const colors = {
        ALLOW: '#53d769',
        VERIFY_OTP: '#ffd166',
        MANDATE_PREPAID: '#ff8c42',
        BLOCK: '#e94560',
      }
      return colors[action] || '#a0a0b8'
    },
    formatAction(action) {
      return action.replaceAll('_', ' ')
    }
  }
}
</script>
