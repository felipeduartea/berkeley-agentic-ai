# Demo Script

This is what to show and say when recording a demo.

---

## Opening (30 seconds)

**Show:** Terminal in TheAgentCompany directory

**Say:**
> "Most agent benchmarks give you a single pass/fail score. That tells you nothing about what actually happened. Did the agent waste time? Take a weird approach? We built a better way to evaluate agents. Let me show you."

---

## Part 1: The Problem (30 seconds)

**Show:** Open `results/` from a previous run (if it exists)

**Say:**
> "Traditional evaluation: Task passed, score 0.75. That's it. But we want to know HOW the agent performed. Was it efficient? Did it make sense? What would a human think? That's what this does."

**Commands:**
```bash
# Show old-style result
cat results/result.json
```

---

## Part 2: Run the Agent (1 minute)

**Say:**
> "Let's run an actual agent on a real task and see what we get."

**Commands:**
```bash
# Set API key (edit yours in)
export OPENAI_API_KEY="your-key"

# Run
python3 run_agent.py
```

**While it runs, say:**
> "The agent is executing right now. It reads the task, uses GPT-4 to plan an approach, executes commands in a container, and logs everything. Takes about 10 seconds."

**Point out the logs as they appear:**
> "See here - every action is logged with timing. Read task, plan strategy, check workspace... 11 actions total in 8 seconds."

---

## Part 3: The Results (2 minutes)

**Say:**
> "Now here's where it gets interesting. Look at the evaluation."

**Show the terminal output:**

Point to each section:

**1. Scores:**
```
Original Score:  0.750
Green Agent:     0.817
Confidence:      0.833
```

**Say:**
> "The basic score was 0.75. But our comprehensive evaluation gives 0.817 with 83% confidence. Why the difference? Let's see."

**2. Breakdown:**
```
Breakdown:
  deterministic: 0.750
  heuristic    : 0.887
  llm_judge    : 0.830
```

**Say:**
> "Three evaluation layers. Deterministic checks - did it pass? Heuristics - behavioral analysis. And LLM judges - what would a human think? Each weighted by confidence."

**3. Metrics:**
```
Top Metrics:
  1. task_efficiency    : 0.950
  2. action_coherence   : 0.750
  3. completion_quality : 0.850
```

**Say:**
> "The agent was super efficient - 0.95. Good quality - 0.85. But coherence was lower at 0.75. Maybe it could have been more focused."

**4. AI Judges:**
```
AI Judges:
  g_eval: 0.900
  deepeval: 0.760
```

**Say:**
> "Two AI judges evaluated it. G-Eval gave it 0.90 - 'completed successfully with logical approach.' DeepEval was more critical at 0.76."

---

## Part 4: The Reports (1 minute)

**Commands:**
```bash
# Show files
ls results/

# Show summary
cat results/REPORT.md
```

**Say:**
> "You get four files. A markdown summary, the raw actions, the full evaluation, and basic results."

**Scroll through REPORT.md:**

**Say:**
> "Here's the summary. Time, actions, scores. The breakdown table. Top metrics. And the AI judge reasoning - look, full explanations of why each score."

**Show one section:**
```markdown
**g_eval**: 0.850
> The task was largely completed successfully with a logical
> and structured approach...
```

**Say:**
> "This is what you actually want - understanding WHY the score is what it is."

---

## Part 5: The Full Analysis (30 seconds)

**Commands:**
```bash
# Quick peek at the full data
cat results/evaluation.json | head -50
```

**Say:**
> "The full JSON has everything. Every metric, every confidence score, every piece of reasoning. You can feed this into your own analysis tools, compare across runs, whatever you need."

---

## Part 6: How It Works (1 minute)

**Show:** Open `run_agent.py` in an editor

**Scroll to the evaluation part (around line 150):**

**Say:**
> "Here's the code. We create the framework with a scoring policy. Then evaluate_task with the trajectory, result, and task description. It returns a score and full report. That's it."

**Show the framework files:**
```bash
ls evaluation/comprehensive_framework/
```

**Say:**
> "The framework has four main parts. Core orchestrates everything. LLM judges uses GPT-4 for G-Eval and DeepEval. Heuristics analyzes behavior. And scoring combines it all with confidence weighting."

---

## Part 7: Why This Matters (1 minute)

**Say:**
> "So why does this matter? Three reasons."

**Show terminal with results still visible:**

**1. Understanding**
> "First, you actually understand what happened. Not just pass/fail, but how efficient, how coherent, how well-reasoned the agent was."

**2. Debugging**
> "Second, you can debug. Low coherence score? Maybe the agent is jumping around too much. Low efficiency? Maybe it's wasting actions."

**3. Comparison**
> "Third, you can meaningfully compare agents. Not just 'this one got 0.75 and this one got 0.80.' But this one is more efficient, this one is more coherent, this one has better AI judge scores."

---

## Part 8: Wrap Up (30 seconds)

**Show:** `QUICK.md` or `README_GREENAGENT.md`

**Say:**
> "Setup is simple. Three pip installs, set your API key, run the script. Takes 10-15 seconds per evaluation, costs about 2 cents in API calls. All the code is here, fully documented."

**Final screen - show the results directory:**

**Say:**
> "That's it. Better evaluation, actual insights, and you can use it right now."

---

## Screen Setup Tips

### Before Recording:

1. **Clean your terminal:**
```bash
```

2. **Remove old results:**
```bash
rm -rf results/
```

3. **Set up a split screen:**
   - Left: Terminal for running commands
   - Right: Editor for showing code/files

4. **Increase font size:**
```bash
# In terminal preferences, use 16-18pt font
```

5. **Use a simple theme:**
   - Light background is better for most videos
   - Or dark with high contrast

### During Recording:

1. **Pause between sections** - gives you edit points

2. **Repeat key points while showing them**

3. **Don't rush the logs** - let people see them scrolling

4. **Use your mouse** - point at specific numbers/lines

5. **Show the files being created** - `ls results/` after the run

### Good B-Roll Shots:

- Code in `run_agent.py` (the simple agent class)
- Framework architecture in `evaluation/comprehensive_framework/`
- A generated report with highlighting
- The evaluation breakdown table
- AI judge reasoning sections

---

## Alternative: Live Demo

If doing this live (not pre-recorded):

1. **Have results pre-generated** as backup
2. **Test your API key** beforehand
3. **Have the docs open** in a browser tab
4. **Practice the timing** - know where to pause
5. **Prepare for questions:**
   - How long does it take? ~10-15s
   - How much does it cost? ~$0.02
   - Can I use my own metrics? Yes, extend BaseMetric
   - What models? GPT-4o for LLM judges

---

## One-Liner Demo

For a super quick demo (30 seconds):

```bash
# Clear
clear

# Run
python3 run_agent.py

# Show results
cat results/REPORT.md
```

**Say:**
> "Run the agent, get comprehensive evaluation. Not just pass/fail but behavioral analysis and AI assessment. Done."

That's the 30-second version.

