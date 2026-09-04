<template>
  <div class="shap-container glass-card">
    <div class="section-header">
      <span class="icon">[+]</span>
      <h2>Explainability (SHAP / Rules)</h2>
    </div>
    
    <div v-if="!reasons || reasons.length === 0" class="empty-state">
      <div class="icon">[i]</div>
      <p>Evaluate a transaction to see why it was scored.</p>
    </div>
    
    <div v-else class="shap-bar-container">
      <div 
        v-for="(reason, index) in sortedReasons" 
        :key="index"
        class="shap-bar-item"
      >
        <div class="shap-feature-name" :title="reason.feature">
          {{ formatFeatureName(reason.feature) }}
        </div>
        
        <div class="shap-bar-track">
          <div 
            class="shap-bar-fill" 
            :class="reason.contribution >= 0 ? 'positive' : 'negative'"
            :style="{ width: getBarWidth(reason.contribution) }"
          ></div>
        </div>
        
        <div class="shap-value" :class="reason.contribution >= 0 ? 'positive' : 'negative'">
          {{ reason.contribution >= 0 ? '+' : '' }}{{ reason.contribution.toFixed(3) }}
        </div>
        
        <div class="shap-description">
          {{ reason.description }}
        </div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'ShapWaterfall',
  props: {
    reasons: {
      type: Array,
      default: () => []
    }
  },
  computed: {
    sortedReasons() {
      // Sort by absolute contribution (most important first)
      return [...this.reasons].sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution))
    },
    maxContribution() {
      if (!this.reasons || this.reasons.length === 0) return 0
      return Math.max(...this.reasons.map(r => Math.abs(r.contribution)))
    }
  },
  methods: {
    formatFeatureName(feature) {
      // Convert snake_case to Title Case
      return feature.split('_').map(word => 
        word.charAt(0).toUpperCase() + word.slice(1)
      ).join(' ')
    },
    getBarWidth(contribution) {
      if (this.maxContribution === 0) return '0%'
      // Calculate width relative to the maximum absolute contribution, capped at 100%
      const width = (Math.abs(contribution) / this.maxContribution) * 100
      return `${Math.min(width, 100)}%`
    }
  }
}
</script>
