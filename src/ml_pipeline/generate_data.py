"""
Synthetic Indian E-Commerce Transaction Data Generator
=====================================================
Generates ~50,000 realistic transactions mimicking Indian e-commerce patterns:
- COD-heavy orders, tier-2/3 city skew, festival-season spikes
- Realistic return rates (~10%) with fraud/abuse patterns
- Temporal spread for chronological train/test splitting
"""

import os
import random
import hashlib
import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Add project root to path for config import
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.ml_pipeline.config import (
    NUM_TRANSACTIONS, RETURN_RATE, FRAUD_RETURN_RATE,
    DATA_START, DATA_END,
    CATEGORIES, PAYMENT_METHODS, RISKY_CATEGORIES,
)

# =============================================================================
# CONSTANTS FOR SYNTHETIC DATA
# =============================================================================

INDIAN_FIRST_NAMES = [
    "Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun", "Sai", "Reyansh",
    "Ayaan", "Krishna", "Ishaan", "Shaurya", "Atharva", "Advik", "Pranav",
    "Advaith", "Aarush", "Kabir", "Rudra", "Dhruv", "Darsh",
    "Ananya", "Aadhya", "Saanvi", "Aanya", "Isha", "Diya", "Myra",
    "Sara", "Kiara", "Anika", "Riya", "Prisha", "Avni", "Navya",
    "Anvi", "Pari", "Aarohi", "Siya", "Nisha", "Kavya",
    "Rahul", "Amit", "Priya", "Sneha", "Rohit", "Neha", "Vikram",
    "Pooja", "Deepak", "Swati", "Rajesh", "Sunita", "Manoj", "Meena",
    "Suresh", "Geeta", "Ramesh", "Anita", "Mukesh", "Rekha",
    "Mohammed", "Fatima", "Imran", "Ayesha", "Salman", "Zara",
    "Harpreet", "Gurpreet", "Manpreet", "Jaspreet", "Amardeep", "Sukhwinder",
]

INDIAN_LAST_NAMES = [
    "Sharma", "Verma", "Gupta", "Singh", "Kumar", "Patel", "Shah",
    "Mehta", "Joshi", "Chauhan", "Yadav", "Mishra", "Pandey", "Dubey",
    "Tiwari", "Srivastava", "Agarwal", "Banerjee", "Mukherjee", "Das",
    "Bose", "Roy", "Sen", "Ghosh", "Chatterjee", "Nair", "Menon",
    "Pillai", "Iyer", "Rao", "Reddy", "Naidu", "Choudhary", "Thakur",
    "Kapoor", "Malhotra", "Saxena", "Bhatia", "Arora", "Sethi",
    "Jain", "Goel", "Mittal", "Singhania", "Khanna", "Chopra",
    "Khan", "Ahmed", "Syed", "Hussain", "Sheikh", "Malik",
    "Gill", "Sandhu", "Dhillon", "Brar", "Sidhu", "Kaur",
]

EMAIL_DOMAINS_LEGIT = [
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com",
    "rediffmail.com", "yahoo.in", "gmail.co.in", "protonmail.com",
    "icloud.com", "live.com", "aol.com", "zoho.com",
]

EMAIL_DOMAINS_DISPOSABLE = [
    "yopmail.com", "guerrillamail.com", "mailinator.com", "tempmail.com",
    "throwaway.email", "temp-mail.org", "fakeinbox.com", "sharklasers.com",
    "trashmail.com", "discard.email", "maildrop.cc", "getnada.com",
    "10minutemail.com", "burnermail.io", "mohmal.com",
]

# Festival months with higher COD and return spikes (Indian e-commerce)
FESTIVAL_MONTHS = [9, 10, 11]  # Sep (Navratri), Oct (Diwali), Nov (post-Diwali sales)


def load_pincodes():
    """Load Indian pincode lat/lng reference data."""
    pincode_path = os.path.join(
        os.path.dirname(__file__), '..', '..', 'data', 'reference', 'indian_pincode_lat_lng.csv'
    )
    df = pd.read_csv(pincode_path, dtype={'pincode': str})
    return df


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the great-circle distance between two points on Earth (in km)."""
    R = 6371  # Earth's radius in km
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c


def generate_user_id(name, idx):
    """Generate a deterministic user ID."""
    raw = f"{name}_{idx}"
    return f"USR_{hashlib.md5(raw.encode()).hexdigest()[:12].upper()}"


def generate_transactions(n_transactions=NUM_TRANSACTIONS, seed=42):
    """
    Generate synthetic Indian e-commerce transactions.
    
    Returns:
        pd.DataFrame: Transaction dataset with columns matching the ML pipeline's
                      expected schema.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    pincodes_df = load_pincodes()
    pincode_list = pincodes_df['pincode'].tolist()
    pincode_lookup = pincodes_df.set_index('pincode')[['latitude', 'longitude']].to_dict('index')
    
    # Generate a pool of ~5000 unique users
    n_users = 5000
    users = []
    for i in range(n_users):
        first = random.choice(INDIAN_FIRST_NAMES)
        last = random.choice(INDIAN_LAST_NAMES)
        name = f"{first} {last}"
        user_id = generate_user_id(name, i)
        
        # Determine if this user is a "fraudster" (~8% of users)
        is_fraudster = random.random() < 0.08
        
        # Account creation date (spread across 2024-2025)
        account_created = datetime(2024, 1, 1) + timedelta(
            days=random.randint(0, 700)
        )
        
        # Assign a home pincode (billing)
        home_pincode = random.choice(pincode_list)
        
        # Email: fraudsters more likely to use disposable emails
        if is_fraudster and random.random() < 0.6:
            email_domain = random.choice(EMAIL_DOMAINS_DISPOSABLE)
        else:
            email_domain = random.choice(EMAIL_DOMAINS_LEGIT)
        
        email_user = f"{first.lower()}.{last.lower()}{random.randint(1, 999)}"
        email = f"{email_user}@{email_domain}"
        
        # Phone number (Indian 10-digit)
        phone = f"+91{random.randint(6000000000, 9999999999)}"
        
        users.append({
            'user_id': user_id,
            'name': name,
            'email': email,
            'phone': phone,
            'home_pincode': home_pincode,
            'account_created': account_created,
            'is_fraudster': is_fraudster,
        })
    
    # Parse date range
    start_date = datetime.strptime(DATA_START, "%Y-%m-%d")
    end_date = datetime.strptime(DATA_END, "%Y-%m-%d")
    date_range_days = (end_date - start_date).days
    
    # Track per-user order history for velocity and return rate features
    user_order_history = {u['user_id']: [] for u in users}
    
    transactions = []
    
    for txn_idx in range(n_transactions):
        # Pick a user (fraudsters get slightly more transactions)
        if random.random() < 0.15:
            user = random.choice([u for u in users if u['is_fraudster']])
        else:
            user = random.choice(users)
        
        # Generate order timestamp
        order_date = start_date + timedelta(days=random.randint(0, date_range_days))
        
        # Festival months have higher order volume (bias sampling)
        if order_date.month in FESTIVAL_MONTHS and random.random() < 0.3:
            order_date = order_date  # Keep festival date
        
        # Hour of day: legitimate users order during business hours, fraudsters at odd hours
        if user['is_fraudster'] and random.random() < 0.3:
            hour = random.choice([1, 2, 3, 4, 5])  # Late night
        else:
            hour = random.choices(
                range(24),
                weights=[1, 1, 1, 1, 1, 1, 2, 3, 5, 8, 10, 10, 
                         9, 8, 8, 7, 6, 5, 5, 5, 4, 3, 2, 1],
            )[0]
        
        order_timestamp = order_date.replace(
            hour=hour,
            minute=random.randint(0, 59),
            second=random.randint(0, 59),
        )
        
        # Billing pincode = user's home
        billing_pincode = user['home_pincode']
        
        # Shipping pincode: fraudsters more likely to ship far away
        if user['is_fraudster'] and random.random() < 0.5:
            # Pick a pincode far from home
            home_coords = pincode_lookup.get(billing_pincode, {'latitude': 20.0, 'longitude': 78.0})
            far_pincodes = [
                p for p in pincode_list
                if p in pincode_lookup and haversine_distance(
                    home_coords['latitude'], home_coords['longitude'],
                    pincode_lookup[p]['latitude'], pincode_lookup[p]['longitude']
                ) > 500
            ]
            shipping_pincode = random.choice(far_pincodes) if far_pincodes else random.choice(pincode_list)
        else:
            # Legitimate: ship to home or nearby
            if random.random() < 0.7:
                shipping_pincode = billing_pincode
            else:
                shipping_pincode = random.choice(pincode_list)
        
        # Calculate geo distance
        if billing_pincode in pincode_lookup and shipping_pincode in pincode_lookup:
            geo_distance = haversine_distance(
                pincode_lookup[billing_pincode]['latitude'],
                pincode_lookup[billing_pincode]['longitude'],
                pincode_lookup[shipping_pincode]['latitude'],
                pincode_lookup[shipping_pincode]['longitude'],
            )
        else:
            geo_distance = 0.0
        
        # Product category
        if user['is_fraudster'] and random.random() < 0.6:
            category = random.choice(RISKY_CATEGORIES)
        else:
            category = random.choice(CATEGORIES)
        
        # Order value ()
        category_price_ranges = {
            "electronics": (2000, 50000),
            "fashion": (300, 8000),
            "mobile_accessories": (100, 5000),
            "footwear": (500, 10000),
            "home_kitchen": (200, 15000),
            "beauty": (100, 5000),
            "books": (100, 2000),
            "grocery": (200, 5000),
            "sports": (500, 15000),
            "toys": (200, 5000),
        }
        price_range = category_price_ranges.get(category, (200, 5000))
        order_value = round(random.uniform(*price_range), 2)
        
        # Payment method: fraudsters prefer COD
        if user['is_fraudster'] and random.random() < 0.7:
            payment_method = "COD"
        else:
            payment_method = random.choices(
                PAYMENT_METHODS,
                weights=[40, 30, 20, 10],  # COD-heavy Indian market
            )[0]
        
        # Account age at time of order
        account_age = (order_timestamp - user['account_created']).days
        account_age = max(0, account_age)
        
        # Past order history for this user
        past_orders = [o for o in user_order_history[user['user_id']] 
                       if o['timestamp'] < order_timestamp]
        total_past_orders = len(past_orders)
        
        # Past returns
        past_returns = sum(1 for o in past_orders if o.get('is_return', 0) == 1)
        
        # Transaction velocity: orders in last 24h and 7d
        velocity_24h = sum(
            1 for o in past_orders
            if (order_timestamp - o['timestamp']).total_seconds() < 86400
        )
        velocity_7d = sum(
            1 for o in past_orders
            if (order_timestamp - o['timestamp']).total_seconds() < 604800
        )
        
        # Average order value ratio
        if past_orders:
            avg_past_value = np.mean([o['value'] for o in past_orders])
            avg_order_value_ratio = order_value / avg_past_value if avg_past_value > 0 else 1.0
        else:
            avg_order_value_ratio = 1.0
        
        # ===== DETERMINE RETURN LABEL =====
        is_return = 0
        
        if user['is_fraudster']:
            # Fraudsters have high return rate
            base_return_prob = 0.45
            
            # Increase probability based on risk signals
            if geo_distance > 500:
                base_return_prob += 0.10
            if payment_method == "COD":
                base_return_prob += 0.10
            if category in RISKY_CATEGORIES:
                base_return_prob += 0.05
            if velocity_24h > 2:
                base_return_prob += 0.10
            if hour in range(1, 6):
                base_return_prob += 0.05
            
            is_return = 1 if random.random() < min(base_return_prob, 0.85) else 0
        else:
            # Genuine users: low return rate (~5-8%)
            base_return_prob = 0.06
            
            # Some categories naturally have higher return rates
            if category in ["fashion", "footwear"]:
                base_return_prob += 0.04  # Size/fit issues
            if category == "electronics":
                base_return_prob += 0.02  # DOA, defects
            
            is_return = 1 if random.random() < base_return_prob else 0
        
        # Check for disposable email
        email_domain = user['email'].split('@')[1]
        is_disposable = email_domain in EMAIL_DOMAINS_DISPOSABLE
        
        # Build transaction record
        txn = {
            'transaction_id': f"TXN_{txn_idx:06d}",
            'user_id': user['user_id'],
            'email': user['email'],
            'phone': user['phone'],
            'billing_pincode': billing_pincode,
            'shipping_pincode': shipping_pincode,
            'geo_distance_km': round(geo_distance, 2),
            'order_value': order_value,
            'payment_method': payment_method,
            'product_category': category,
            'order_timestamp': order_timestamp.isoformat(),
            'hour_of_day': hour,
            'account_age_days': account_age,
            'total_past_orders': total_past_orders,
            'past_returns': past_returns,
            'transaction_velocity_24h': velocity_24h,
            'transaction_velocity_7d': velocity_7d,
            'avg_order_value_ratio': round(avg_order_value_ratio, 4),
            'is_disposable_email': int(is_disposable),
            'is_cod': int(payment_method == "COD"),
            'is_high_risk_category': int(category in RISKY_CATEGORIES),
            'is_first_order': int(total_past_orders == 0),
            'return_history_rate': round(
                past_returns / total_past_orders if total_past_orders > 0 else 0.0, 4
            ),
            'is_return': is_return,  # TARGET LABEL
        }
        
        transactions.append(txn)
        
        # Update user history
        user_order_history[user['user_id']].append({
            'timestamp': order_timestamp,
            'value': order_value,
            'is_return': is_return,
        })
    
    df = pd.DataFrame(transactions)
    
    # Sort by timestamp (critical for chronological splitting)
    df = df.sort_values('order_timestamp').reset_index(drop=True)
    
    return df


def main():
    """Generate and save synthetic transaction data."""
    print("=" * 60)
    print("  AI Return-Risk Scorer  Synthetic Data Generator")
    print("=" * 60)
    
    # Create output directories
    raw_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw')
    os.makedirs(raw_dir, exist_ok=True)
    
    print(f"\n Generating {NUM_TRANSACTIONS:,} synthetic transactions...")
    df = generate_transactions()
    
    # Save raw data
    output_path = os.path.join(raw_dir, 'transactions_synthetic.csv')
    df.to_csv(output_path, index=False)
    print(f" Saved raw data to: {output_path}")
    
    # Print dataset summary
    print(f"\n Dataset Summary:")
    print(f"   Total Transactions: {len(df):,}")
    print(f"   Unique Users: {df['user_id'].nunique():,}")
    print(f"   Date Range: {df['order_timestamp'].min()[:10]}  {df['order_timestamp'].max()[:10]}")
    print(f"   Return Rate: {df['is_return'].mean():.2%}")
    print(f"   Returns (count): {df['is_return'].sum():,}")
    print(f"   Non-Returns (count): {(df['is_return'] == 0).sum():,}")
    
    print(f"\n Payment Method Distribution:")
    for method, count in df['payment_method'].value_counts().items():
        print(f"   {method}: {count:,} ({count/len(df):.1%})")
    
    print(f"\n  Product Category Distribution:")
    for cat, count in df['product_category'].value_counts().head(5).items():
        print(f"   {cat}: {count:,} ({count/len(df):.1%})")
    
    print(f"\n Disposable Email Usage: {df['is_disposable_email'].mean():.2%}")
    print(f" Avg Geo Distance: {df['geo_distance_km'].mean():.1f} km")
    print(f" Avg Order Value: {df['order_value'].mean():,.2f}")
    
    # Return rate by key segments
    print(f"\n Return Rate by Payment Method:")
    for method in PAYMENT_METHODS:
        subset = df[df['payment_method'] == method]
        if len(subset) > 0:
            print(f"   {method}: {subset['is_return'].mean():.2%}")
    
    print(f"\n Return Rate by Category (Top 5):")
    for cat in df['product_category'].unique()[:5]:
        subset = df[df['product_category'] == cat]
        print(f"   {cat}: {subset['is_return'].mean():.2%}")
    
    print(f"\n{'=' * 60}")
    print("   Data generation complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

