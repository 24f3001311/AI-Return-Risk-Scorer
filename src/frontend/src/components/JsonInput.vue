<template>
  <div class="json-input-wrapper glass-card">
    <div class="section-header">
      <span class="icon">📝</span>
      <h2>Transaction Payload</h2>
    </div>
    
    <textarea
      ref="textareaRef"
      class="json-textarea"
      :class="{ error: jsonError }"
      v-model="jsonText"
      placeholder='Paste your transaction JSON here...'
      spellcheck="false"
    ></textarea>
    
    <div v-if="jsonError" class="json-error">
      ⚠️ {{ jsonError }}
    </div>
    
    <div class="evaluate-actions">
      <button 
        class="btn btn-primary" 
        @click="handleEvaluate"
        :disabled="loading || !!jsonError"
        id="evaluate-btn"
      >
        <span v-if="loading" class="loading-spinner"></span>
        <span v-else>🔍</span>
        {{ loading ? 'Evaluating...' : 'Evaluate Risk' }}
      </button>
      
      <button class="btn btn-secondary" @click="loadSample" id="sample-btn">
        📋 Load Sample
      </button>
      
      <button class="btn btn-secondary" @click="loadRiskySample" id="risky-sample-btn">
        ⚠️ Risky Sample
      </button>
      
      <button class="btn btn-secondary" @click="clearInput" id="clear-btn">
        🗑️ Clear
      </button>
    </div>
  </div>
</template>

<script>
function generateSafeSample() {
  const randomSuffix = Math.floor(Math.random() * 1000);
  const totalOrders = Math.floor(Math.random() * 30) + 5; // 5 to 34
  return {
    user_id: `USR_SAFE${randomSuffix}`,
    email: `customer${randomSuffix}@gmail.com`,
    phone: `+9198765${String(randomSuffix).padStart(5, '0')}`,
    billing_pincode: "110001",
    shipping_pincode: "110005", // close distance
    order_value: Math.floor(Math.random() * 3000) + 500,
    payment_method: Math.random() > 0.5 ? "UPI" : "CARD",
    product_category: "books",
    account_age_days: Math.floor(Math.random() * 500) + 100,
    total_past_orders: totalOrders,
    past_returns: Math.floor(totalOrders * (Math.random() * 0.1)), // 0-10% return rate
    transaction_velocity_24h: 1,
    transaction_velocity_7d: Math.floor(Math.random() * 3) + 1
  };
}

function generateRiskySample() {
  const randomSuffix = Math.floor(Math.random() * 1000);
  return {
    user_id: `USR_FRAUD${randomSuffix}`,
    email: `random${randomSuffix}@yopmail.com`,
    phone: `+9160000${String(randomSuffix).padStart(5, '0')}`,
    billing_pincode: "110001",
    shipping_pincode: "600001", // high distance
    order_value: Math.floor(Math.random() * 20000) + 15000,
    payment_method: "COD",
    product_category: "electronics",
    account_age_days: Math.floor(Math.random() * 3), // 0-2 days
    total_past_orders: 0,
    past_returns: 0,
    transaction_velocity_24h: Math.floor(Math.random() * 3) + 3, // 3-5 orders in 24h
    transaction_velocity_7d: Math.floor(Math.random() * 4) + 5 // 5-8 orders in 7d
  };
}

export default {
  name: 'JsonInput',
  props: {
    loading: {
      type: Boolean,
      default: false
    }
  },
  emits: ['evaluate'],
  data() {
    return {
      jsonText: JSON.stringify(generateSafeSample(), null, 2),
      jsonError: null,
    }
  },
  watch: {
    jsonText() {
      this.validateJson()
    }
  },
  methods: {
    validateJson() {
      try {
        if (this.jsonText.trim()) {
          JSON.parse(this.jsonText)
          this.jsonError = null
        }
      } catch (e) {
        this.jsonError = `Invalid JSON: ${e.message}`
      }
    },
    handleEvaluate() {
      if (this.jsonError) return
      try {
        const payload = JSON.parse(this.jsonText)
        this.$emit('evaluate', payload)
      } catch (e) {
        this.jsonError = `Parse error: ${e.message}`
      }
    },
    loadSample() {
      this.jsonText = JSON.stringify(generateSafeSample(), null, 2)
      this.jsonError = null
    },
    loadRiskySample() {
      this.jsonText = JSON.stringify(generateRiskySample(), null, 2)
      this.jsonError = null
    },
    clearInput() {
      this.jsonText = ''
      this.jsonError = null
    }
  },
  mounted() {
    this.validateJson()
  }
}
</script>
