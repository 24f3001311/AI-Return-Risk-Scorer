"""
Pydantic Schema  Transaction Payload (Input)

Strictly validates incoming JSON payloads from merchant systems.
Invalid payloads return 422 Unprocessable Entity.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Literal


class TransactionPayload(BaseModel):
    """
    Incoming transaction payload from the merchant's system.
    
    All fields are required and validated. The API will return a 
    422 error with detailed field-level error messages if validation fails.
    """
    
    user_id: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Unique identifier for the customer",
        examples=["USR_A1B2C3D4E5F6"],
    )
    
    email: str = Field(
        ...,
        min_length=5,
        max_length=100,
        description="Customer's email address",
        examples=["rahul.sharma42@gmail.com"],
    )
    
    phone: str = Field(
        ...,
        min_length=10,
        max_length=15,
        description="Customer's phone number (Indian format)",
        examples=["+919876543210"],
    )
    
    billing_pincode: str = Field(
        ...,
        min_length=6,
        max_length=6,
        pattern=r"^\d{6}$",
        description="6-digit Indian billing pincode",
        examples=["110001"],
    )
    
    shipping_pincode: str = Field(
        ...,
        min_length=6,
        max_length=6,
        pattern=r"^\d{6}$",
        description="6-digit Indian shipping pincode",
        examples=["400001"],
    )
    
    order_value: float = Field(
        ...,
        gt=0,
        le=500000,
        description="Order value in INR ()",
        examples=[2499.99],
    )
    
    payment_method: Literal["COD", "UPI", "CARD", "WALLET"] = Field(
        ...,
        description="Payment method used for the order",
        examples=["COD"],
    )
    
    product_category: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Product category of the order",
        examples=["electronics"],
    )
    
    account_age_days: int = Field(
        ...,
        ge=0,
        le=3650,
        description="Age of the customer's account in days",
        examples=[45],
    )
    
    total_past_orders: int = Field(
        ...,
        ge=0,
        le=10000,
        description="Total number of past orders by this customer",
        examples=[12],
    )
    
    past_returns: int = Field(
        ...,
        ge=0,
        le=10000,
        description="Total number of past returns by this customer",
        examples=[2],
    )
    
    transaction_velocity_24h: int = Field(
        ...,
        ge=0,
        le=100,
        description="Number of orders by this customer in the last 24 hours",
        examples=[1],
    )
    
    transaction_velocity_7d: int = Field(
        ...,
        ge=0,
        le=500,
        description="Number of orders by this customer in the last 7 days",
        examples=[3],
    )
    
    @field_validator('email')
    @classmethod
    def validate_email(cls, v):
        """Basic email format validation."""
        if '@' not in v or '.' not in v.split('@')[-1]:
            raise ValueError('Invalid email format')
        return v.lower()
    
    @field_validator('past_returns')
    @classmethod
    def validate_returns_not_exceed_orders(cls, v, info):
        """Ensure past_returns  total_past_orders."""
        if 'total_past_orders' in info.data and v > info.data['total_past_orders']:
            raise ValueError('past_returns cannot exceed total_past_orders')
        return v

