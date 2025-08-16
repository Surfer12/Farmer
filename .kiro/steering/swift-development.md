---
inclusion: manual
---
# Swift Development Guidelines

## Code Organization

### iOS App Structure
- [Farmer/](mdc:Farmer/) - Main iOS app code
- [FarmerTests/](mdc:FarmerTests/) - Unit tests
- [FarmerUITests/](mdc:FarmerUITests/) - UI tests

### Swift Package Structure  
- [Sources/UOIFCLI/](mdc:Sources/UOIFCLI/) - Command line interface
- [Sources/UOIFCore/](mdc:Sources/UOIFCore/) - Core framework
- [Tests/UOIFCoreTests/](mdc:Tests/UOIFCoreTests/) - Package tests

## Swift Style Guidelines

### Naming Conventions
- Use descriptive names: functions are verbs, properties are nouns
- Follow Swift naming conventions (camelCase for variables/functions)
- Use meaningful type names that describe their purpose

### Code Structure
- Prefer structs over classes when possible (value semantics)
- Use protocols for abstraction and testability
- Keep functions focused and single-purpose
- Use guard statements for early exits

### SwiftUI Best Practices
- Separate view logic from business logic
- Use @StateObject for owned objects, @ObservedObject for injected
- Keep view hierarchies shallow and composable

### Testing
- Write unit tests for business logic
- Use dependency injection for testability
- Mock external dependencies in tests

## Project Configuration

### Build Configuration
- [Package.swift](mdc:Package.swift) - Swift package manifest
- Xcode project settings in [Farmer.xcodeproj/](mdc:Farmer.xcodeproj/)

### Dependencies
- Prefer Swift Package Manager over other dependency managers
- Keep dependencies minimal and well-maintained
- Document dependency choices in commit messages