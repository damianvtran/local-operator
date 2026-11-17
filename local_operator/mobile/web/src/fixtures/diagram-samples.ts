/**
 * One small, valid source per diagram kind the relay may be asked to draw.
 *
 * Shared by the unit tests (every kind must survive the SVG sanitiser with its
 * shapes intact) and by the browser rig. The three phone-width cases the PR
 * evidence shows (flowchart, sequence, state) are the first three.
 */
export const DIAGRAM_SAMPLES: Record<string, string> = {
	flowchart: `flowchart TD
    A[Operator sends a prompt] --> B{Plane reachable?}
    B -->|yes| C[Stream the turn]
    B -->|no| D[Show the offline banner]
    C --> E[Render the transcript]
    D --> F[Retry with backoff]
    F --> B
    E --> G((Done))
`,
	sequence: `sequenceDiagram
    participant U as Operator
    participant UI as Phone
    participant P as Relay
    U->>UI: Send prompt
    UI->>P: POST /turns
    P-->>UI: Stream events
    loop Every event
        UI->>UI: Reduce into transcript
    end
    UI-->>U: Settled answer
`,
	state: `stateDiagram-v2
    [*] --> Idle
    Idle --> Streaming: send
    Streaming --> Settled: done
    Streaming --> Failed: error
    Failed --> Idle: retry
    Settled --> Idle: next prompt
    Settled --> [*]
`,
	class: `classDiagram
    class Session {
      +String id
      +send(text) Turn
    }
    class Turn
    Session "1" --> "*" Turn : holds
    Session <|-- TemporarySession
`,
	er: `erDiagram
    SESSION ||--o{ TURN : holds
    TURN ||--|{ ROW : renders
    SESSION {
      string id PK
      string title
    }
`,
	journey: `journey
    title A change through the gate
    section Spec
      Scope the slice: 5: Operator
      Measure the floors: 3: Operator, Agent
    section Build
      Implement: 4: Agent
`,
	gantt: `gantt
    title Release
    dateFormat YYYY-MM-DD
    section Build
    Code :a1, 2026-01-01, 5d
    Review :after a1, 3d
`,
	pie: `pie title Turn time
    "Model" : 62
    "Tools" : 25
    "Render" : 8
    "Other" : 5
`,
	quadrant: `quadrantChart
    title Reach and engagement
    x-axis Low Reach --> High Reach
    y-axis Low Engagement --> High Engagement
    quadrant-1 Expand
    quadrant-2 Promote
    quadrant-3 Re-evaluate
    quadrant-4 Improve
    Campaign A: [0.3, 0.6]
    Campaign B: [0.45, 0.23]
`,
	requirement: `requirementDiagram
    requirement test_req {
      id: 1
      text: the test text.
      risk: high
      verifymethod: test
    }
    element test_entity {
      type: simulation
    }
    test_entity - satisfies -> test_req
`,
	gitGraph: `gitGraph
    commit
    branch develop
    commit
    checkout main
    merge develop
`,
	mindmap: `mindmap
  root((Relay))
    Transcript
      Markdown
      Diagrams
    Composer
`,
	timeline: `timeline
    title History
    2024 : Alpha
    2025 : Beta : Launch
`,
	sankey: `sankey-beta
A,B,10
A,C,5
B,D,8
`,
	xychart: `xychart-beta
    title "Sales"
    x-axis [jan, feb, mar]
    y-axis "Revenue" 0 --> 100
    bar [20, 50, 80]
    line [20, 50, 80]
`,
	block: `block-beta
  columns 3
  a["A"] b["B"] c["C"]
  a --> b
`,
	packet: `packet-beta
  0-15: "Source Port"
  16-31: "Destination Port"
`,
	kanban: `kanban
  todo[Todo]
    t1[Write the tests]
  done[Done]
    t2[Ship]
`,
	architecture: `architecture-beta
    group api(cloud)[API]
    service db(database)[Database] in api
    service srv(server)[Server] in api
    db:L -- R:srv
`,
	c4: `C4Context
    title System context
    Person(user, "Operator")
    System(sys, "Relay")
    Rel(user, sys, "Uses")
`,
	radar: `radar-beta
  axis a["Speed"], b["Cost"], c["Quality"]
  curve x["Plan"]{3, 2, 4}
  max 5
`,
	treemap: `treemap-beta
"Section"
    "Leaf A": 20
    "Leaf B": 30
`,
};
