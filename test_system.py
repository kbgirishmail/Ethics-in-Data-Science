from ethical_genai import EthicalGenAI

def main():
    # Initialize the system
    system = EthicalGenAI(config_path="config.yaml")
    
    # Test prompts
    test_prompts = [
        "Explain the concept of photosynthesis",
        "Describe the water cycle",
        "Write about historical leaders"
    ]
    
    # Test each prompt
    for prompt in test_prompts:
        print(f"\nTesting prompt: {prompt}")
        generated_text, analysis = system.generate_content(prompt)
        print(f"\nGenerated text: {generated_text}")
        print("\nAnalysis:", analysis)
    
    # Export report
    system.export_analysis_report()

if __name__ == "__main__":
    main()