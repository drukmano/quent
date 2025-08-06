#!/usr/bin/env python3
"""
Clean test of quent's exception formatting with deep, complex stack traces.
"""

from quent import Chain


def level_1(x):
    """Entry point - regular function"""
    return level_2(x * 2)


def level_2(x):
    """Regular function that creates a chain"""
    result = Chain(x).then(level_3).then(level_4)()
    return result


def level_3(x):
    """Chain step that modifies value"""
    return x + 100


def level_4(x):
    """Chain step that calls regular function"""
    return level_5(x / 2)


def level_5(x):
    """Regular function with nested chain"""
    chain = (Chain(x)
             .then(lambda v: v * 3)
             .then(level_6)
             .then(level_7))
    return chain()


def level_6(x):
    """Chain step with computation"""
    return x ** 2


def level_7(x):
    """Chain step calling another regular function"""
    return level_8(x)


def level_8(x):
    """Regular function with another chain"""
    result = Chain(x).then(level_9)()
    return level_10(result)


def level_9(x):
    """Chain step with validation"""
    if x > 1000000:
        return level_11(x)
    return x


def level_10(x):
    """Regular function bypass"""
    return level_11(x * 1.5)


def level_11(x):
    """Another regular function with chain"""
    chain = Chain(x).then(level_12).then(lambda v: level_13(v))
    return chain()


def level_12(x):
    """Chain step"""
    return x / 100


def level_13(x):
    """Regular function calling chain"""
    result = (Chain(x)
              .then(level_14)
              .then(level_15))()
    return result


def level_14(x):
    """Chain step - deep in the stack"""
    return x + 50000


def level_15(x):
    """Final chain step - error occurs here"""
    # This is 15 levels deep - cause an error
    return x / 0  # ZeroDivisionError


def test_deep_exception():
    """Test 1: Simple deep stack trace"""
    print("TEST 1: Deep nested chain (15+ levels)")
    print("-" * 40)
    try:
        result = level_1(10)
        print(f"Result: {result}")
    except Exception as e:
        import traceback
        traceback.print_exc()


def test_mixed_chain_regular():
    """Test 2: Mixed chain and regular calls with different error"""
    
    def process_data(value):
        # Regular function
        return Chain(value).then(validate_range)()
    
    def validate_range(value):
        # Chain step
        if value < 0:
            raise ValueError(f"Negative value not allowed: {value}")
        return transform_data(value)
    
    def transform_data(value):
        # Regular function with nested chain
        result = (Chain(value)
                  .then(lambda x: x * 10)
                  .then(Chain().then(apply_business_logic))
                  .then(format_output))()
        return result
    
    def apply_business_logic(value):
        # Chain step
        return access_database(value / 2)
    
    def access_database(value):
        # Regular function
        data = fetch_record(value)
        return Chain(data).then(process_record)()
    
    def fetch_record(record_id):
        # Regular function
        return {"id": record_id, "data": "test"}
    
    def process_record(record):
        # Chain step - error here
        return record["missing_field"]  # KeyError
    
    def format_output(value):
        # Chain step (won't be reached)
        return str(value)
    
    print("\nTEST 2: Mixed chain/regular with KeyError")
    print("-" * 40)
    try:
        result = process_data(42)
        print(f"Result: {result}")
    except Exception as e:
        import traceback
        traceback.print_exc()


def test_exception_in_lambda():
    """Test 3: Exception in lambda within deep chain"""
    
    def start_processing(x):
        return Chain(x).then(step_one)()
    
    def step_one(x):
        result = (Chain(x)
                  .then(lambda v: v * 2)
                  .then(step_two)
                  .then(lambda v: step_three(v + 100)))()
        return result
    
    def step_two(x):
        return x / 5
    
    def step_three(x):
        chain = (Chain(x)
                 .then(lambda v: v ** 0.5)
                 .then(lambda v: process_value(v))
                 .then(lambda v: v["key"]))()  # This lambda will cause KeyError
        return chain
    
    def process_value(x):
        # Returns a number, not a dict
        return x * 1000
    
    print("\nTEST 3: Exception in lambda expression")
    print("-" * 40)
    try:
        result = start_processing(50)
        print(f"Result: {result}")
    except Exception as e:
        import traceback
        traceback.print_exc()


def test_custom_exception():
    """Test 4: Custom exception thrown deep in chain"""
    
    class DataValidationError(Exception):
        pass
    
    class BusinessLogicError(Exception):
        pass
    
    def process_request(request_id):
        return Chain(request_id).then(load_request)()
    
    def load_request(req_id):
        data = {"id": req_id, "amount": 1000}
        return Chain(data).then(validate_request).then(authorize_request)()
    
    def validate_request(data):
        if data["amount"] > 500:
            return Chain(data).then(check_special_rules)()
        return data
    
    def check_special_rules(data):
        result = apply_rule_engine(data)
        return Chain(result).then(verify_compliance)()
    
    def apply_rule_engine(data):
        # Simulate rule processing
        data["risk_score"] = 85
        return data
    
    def verify_compliance(data):
        if data["risk_score"] > 80:
            raise DataValidationError(f"High risk transaction: score={data['risk_score']}")
        return data
    
    def authorize_request(data):
        return Chain(data).then(final_check)()
    
    def final_check(data):
        raise BusinessLogicError("Authorization system offline")
    
    print("\nTEST 4: Custom exception in deep chain")
    print("-" * 40)
    try:
        result = process_request(12345)
        print(f"Result: {result}")
    except Exception as e:
        import traceback
        traceback.print_exc()


def main():
    """Run all tests"""
    print("=" * 50)
    print("QUENT EXCEPTION FORMATTING TEST")
    print("Testing deep, complex exception stack traces")
    print("=" * 50)
    
    test_deep_exception()
    test_mixed_chain_regular()
    test_exception_in_lambda()
    test_custom_exception()
    
    print("\n" + "=" * 50)
    print("All tests complete")
    print("=" * 50)


if __name__ == "__main__":
    main()
