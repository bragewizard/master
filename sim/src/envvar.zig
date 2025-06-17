
//! Environment Variables for SpikeNet
//! Contains configuration settings for the neural network

const std = @import("std");

/// Environment variables and configuration settings for SpikeNet
pub const EnvVar = struct {
    // Network configuration
    pub var NB_SPIKENET: usize = 5;
    pub var LAYER_CONVERGE: usize = 3;
    pub var SIZE_RESP_ZONE: i32 = 10;
    pub var TIME_STOP: usize = 20;
    pub var STEP: usize = 1;
    pub var RESCUE: usize = 0;
    pub var CONVERGE: usize = 0;
    
    // Display options
    pub var DISPLAY: bool = true;
    pub var FOND: bool = true;
    pub var NO_SHOW: bool = false;
    pub var WAIT_ENTER: bool = false;
    pub var CHRONO: bool = true;
    pub var TESTRES: bool = false;
    
    // Image dimensions
    pub var SIZEX: usize = 320;
    pub var SIZEY: usize = 240;
    
    // Learning options
    pub var LEARN: bool = false;
    pub var LEARN_SUPERV: bool = false;
    pub var WATCH_ACTIVITY: bool = false;
    pub var USE_MULTISCALE: bool = true;
    
    // Save directory
    pub var DIR_SAVE_REBUILD: []const u8 = "output";
    
    /// Initialize environment variables from a configuration file
    pub fn init(fileName: []const u8) !void {
        std.debug.print("Loading environment variables from {s}\n", .{fileName});
        
        // Implementation would parse the configuration file
        // and set the above variables accordingly
        
        // Example of reading a configuration file:
        const file = std.fs.cwd().openFile(fileName, .{}) catch |err| {
            std.debug.print("Error opening config file: {}\n", .{err});
            return err;
        };
        defer file.close();
        
        // Parse the file contents
        var buffer: [1024]u8 = undefined;
        var reader = file.reader();
        
        while (reader.readUntilDelimiterOrEof(&buffer, '\n') catch return error.ReadError) |line| {
            // Skip empty lines and comments
            if (line.len == 0 or line[0] == '#' or line[0] == '/') {
                continue;
            }
            
            // Parse each line as a key-value pair
            if (std.mem.indexOf(u8, line, "=")) |pos| {
                const key = std.mem.trim(u8, line[0..pos], " \t");
                const value = std.mem.trim(u8, line[pos+1..], " \t");
                
                // Set corresponding environment variable
                try parseAndSetEnvVar(key, value);
            }
        }
    }
    
    /// Parse and set an environment variable based on a key-value pair
    fn parseAndSetEnvVar(key: []const u8, value: []const u8) !void {
        if (std.mem.eql(u8, key, "NB_SPIKENET")) {
            NB_SPIKENET = try std.fmt.parseInt(usize, value, 10);
        } else if (std.mem.eql(u8, key, "LAYER_CONVERGE")) {
            LAYER_CONVERGE = try std.fmt.parseInt(usize, value, 10);
        } else if (std.mem.eql(u8, key, "SIZE_RESP_ZONE")) {
            SIZE_RESP_ZONE = try std.fmt.parseInt(i32, value, 10);
        } else if (std.mem.eql(u8, key, "TIME_STOP")) {
            TIME_STOP = try std.fmt.parseInt(usize, value, 10);
        } else if (std.mem.eql(u8, key, "STEP")) {
            STEP = try std.fmt.parseInt(usize, value, 10);
        } else if (std.mem.eql(u8, key, "RESCUE")) {
            RESCUE = try std.fmt.parseInt(usize, value, 10);
        } else if (std.mem.eql(u8, key, "CONVERGE")) {
            CONVERGE = try std.fmt.parseInt(usize, value, 10);
        } else if (std.mem.eql(u8, key, "DISPLAY")) {
            DISPLAY = std.mem.eql(u8, value, "YES") or std.mem.eql(u8, value, "true");
        } else if (std.mem.eql(u8, key, "FOND")) {
            FOND = std.mem.eql(u8, value, "YES") or std.mem.eql(u8, value, "true");
        } else if (std.mem.eql(u8, key, "NO_SHOW")) {
            NO_SHOW = std.mem.eql(u8, value, "YES") or std.mem.eql(u8, value, "true");
        } else if (std.mem.eql(u8, key, "WAIT_ENTER")) {
            WAIT_ENTER = std.mem.eql(u8, value, "YES") or std.mem.eql(u8, value, "true");
        } else if (std.mem.eql(u8, key, "CHRONO")) {
            CHRONO = std.mem.eql(u8, value, "YES") or std.mem.eql(u8, value, "true");
        } else if (std.mem.eql(u8, key, "TESTRES")) {
            TESTRES = std.mem.eql(u8, value, "YES") or std.mem.eql(u8, value, "true");
        } else if (std.mem.eql(u8, key, "LEARN")) {
            LEARN = std.mem.eql(u8, value, "YES") or std.mem.eql(u8, value, "true");
        } else if (std.mem.eql(u8, key, "LEARN_SUPERV")) {
            LEARN_SUPERV = std.mem.eql(u8, value, "YES") or std.mem.eql(u8, value, "true");
        } else if (std.mem.eql(u8, key, "WATCH_ACTIVITY")) {
            WATCH_ACTIVITY = std.mem.eql(u8, value, "YES") or std.mem.eql(u8, value, "true");
        } else if (std.mem.eql(u8, key, "USE_MULTISCALE")) {
            USE_MULTISCALE = std.mem.eql(u8, value, "YES") or std.mem.eql(u8, value, "true");
        } else if (std.mem.eql(u8, key, "DIR_SAVE_REBUILD")) {
            DIR_SAVE_REBUILD = value;
        } else if (std.mem.eql(u8, key, "SIZEX")) {
            SIZEX = try std.fmt.parseInt(usize, value, 10);
        } else if (std.mem.eql(u8, key, "SIZEY")) {
            SIZEY = try std.fmt.parseInt(usize, value, 10);
        }
        // Add more variables as needed
    }
};
