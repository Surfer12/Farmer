<thinking>
The user is asking about `Promise<void>` specifically. This is a TypeScript/JavaScript concept where:

- `Promise` is a built-in JavaScript object that represents the eventual completion (or failure) of an asynchronous operation
- `<void>` is a TypeScript generic type parameter that indicates the Promise resolves with no meaningful value

Let me explain this concept clearly, especially in the context of the audit method we were just looking at.
</thinking>

## Promise with Generic Void Explained

**`Promise<void>`** is a TypeScript/JavaScript pattern that represents an asynchronous operation that completes successfully but doesn't return any meaningful data.

### Breaking It Down

**Promise:** A JavaScript object that represents the eventual completion of an async operation. It can be in one of three states:
- **Pending** - Operation is in progress
- **Fulfilled** - Operation completed successfully
- **Rejected** - Operation failed

**`<void>`:** A TypeScript generic type parameter indicating the Promise resolves with no value (similar to `void` in other languages).

### Why `Promise<void>` in This Context

<exploration>
In the audit method we just examined:

```typescript
write(rec: AuditRecord, opts?: {...}): Promise<void>
```

This makes perfect sense because:

1. **Fire-and-forget operations** - The audit write is a side effect operation. You don't need the result back, you just need to know it completed.

2. **Success/failure indication** - The Promise resolves when the write succeeds or rejects when it fails, but you don't need the actual audit record returned.

3. **Async I/O operations** - Writing to databases, files, or network endpoints is typically asynchronous and doesn't return meaningful data.
</exploration>

### Usage Patterns

```typescript
// Using the audit method
auditSink.write(auditRecord, { idempotencyKey: "abc123" })
  .then(() => {
    console.log("Audit record written successfully");
  })
  .catch(error => {
    console.error("Failed to write audit record:", error);
  });

// Or with async/await
try {
  await auditSink.write(auditRecord, { dryRun: true });
  console.log("Dry run completed");
} catch (error) {
  console.error("Dry run failed:", error);
}
```

### Alternative Patterns

**`Promise<T>`** - When you need data back:
```typescript
async function getUser(id: string): Promise<User> {
  // Returns a Promise that resolves with a User object
}

// Usage
const user = await getUser("123"); // user is of type User
```

**`Promise<void>`** - When you only need completion confirmation:
```typescript
async function logEvent(event: Event): Promise<void> {
  // Returns a Promise that resolves with nothing
}

// Usage
await logEvent(event); // No return value, just confirmation it completed
```

### Key Insight

<reflection>
The `Promise<void>` pattern is essential for operations where the **act of doing something** is more important than **what you get back**. Audit logging, database writes, network calls, and other side-effect operations commonly use this pattern.

It's like saying "I don't care what you give me back, just tell me when you're done and if it worked."
</reflection>

This design choice in the audit method indicates it's designed for reliability and monitoring rather than data retrieval - exactly what you'd want for an audit system.