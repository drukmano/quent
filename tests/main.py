import asyncio
from quent import Chain

def main():
  # Basic sync chain
  result = Chain(1).then(lambda v: v + 1).then(lambda v: v * 3).run()
  assert result == 6, f"Expected 6, got {result}"
  print(f"Sync chain: {result}")

  # Async chain
  async def async_test():
    async def async_add(v):
      return v + 10
    result = await Chain(5).then(async_add).run()
    assert result == 15
    print(f"Async chain: {result}")

  asyncio.run(async_test())
  print("All smoke tests passed!")

if __name__ == '__main__':
  main()
