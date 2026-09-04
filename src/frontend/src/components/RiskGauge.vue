<template>
  <div class="risk-gauge-container glass-card">
    <div class="section-header" style="justify-content: center;">
      <span class="icon">🎯</span>
      <h2>Risk Score</h2>
    </div>
    
    <svg class="gauge-svg" viewBox="0 0 200 110">
      <!-- Background arc -->
      <path
        class="gauge-bg"
        d="M 20 100 A 80 80 0 0 1 180 100"
      />
      <!-- Filled arc -->
      <path
        ref="gaugeFill"
        class="gauge-fill"
        d="M 20 100 A 80 80 0 0 1 180 100"
        :stroke="gaugeColor"
        :stroke-dasharray="arcLength"
        :stroke-dashoffset="arcOffset"
      />
    </svg>
    
    <div class="gauge-score" :style="{ color: gaugeColor }">
      {{ displayScore }}
    </div>
    <div class="gauge-label">Return Risk Score</div>
    
    <!-- Action Badge -->
    <div v-if="action" class="action-badge" :class="action" :id="'action-badge-' + action">
      <span>{{ actionIcon }}</span>
      {{ actionLabel }}
    </div>
    
    <!-- Routing Badge -->
    <div v-if="routing" class="routing-badge" :class="routing">
      <span>{{ routing === 'ml_pipeline' ? '🧠' : '📏' }}</span>
      {{ routing === 'ml_pipeline' ? 'ML Pipeline' : 'Rule Engine' }}
    </div>
  </div>
</template>

<script>
export default {
  name: 'RiskGauge',
  props: {
    score: {
      type: Number,
      default: 0
    },
    action: {
      type: String,
      default: null
    },
    routing: {
      type: String,
      default: null
    }
  },
  computed: {
    displayScore() {
      return this.score.toFixed(4)
    },
    gaugeColor() {
      if (this.score < 0.3) return '#53d769'
      if (this.score < 0.6) return '#ffd166'
      if (this.score < 0.8) return '#ff8c42'
      return '#e94560'
    },
    arcLength() {
      // Total arc length for the semicircle
      const totalLength = Math.PI * 80 // πr for semicircle
      return `${totalLength} ${totalLength}`
    },
    arcOffset() {
      const totalLength = Math.PI * 80
      const offset = totalLength * (1 - this.score)
      return offset
    },
    actionIcon() {
      const icons = {
        ALLOW: '✅',
        VERIFY_OTP: '🔐',
        MANDATE_PREPAID: '💳',
        BLOCK: '🚫',
      }
      return icons[this.action] || '❓'
    },
    actionLabel() {
      const labels = {
        ALLOW: 'Allow Order',
        VERIFY_OTP: 'Verify via OTP',
        MANDATE_PREPAID: 'Mandate Prepaid',
        BLOCK: 'Block Order',
      }
      return labels[this.action] || this.action
    }
  }
}
</script>
